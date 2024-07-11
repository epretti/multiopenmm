# This file is part of MultiOpenMM.
# ©2024 The Regents of the University of California.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
A simple client-server task library.
"""

import atexit
import contextlib
import enum
import functools
import hashlib
import hmac
import itertools
import os
import pickle
import secrets
import select
import socket
import struct
import sys
import tempfile
import time
import traceback

from . import support

_DEFAULT_CONNECT_INTERVAL = 1
_DEFAULT_RUN_INTERVAL = None

LOG = False
DEBUG = False

def _log(*args, **kwargs):
    if LOG:
        print(f"{__name__}:", *args, **kwargs, file=sys.stderr)

def _debug(*args, **kwargs):
    if DEBUG:
        _log("[debug]:",  *args, **kwargs)

class _Kind(enum.Enum):
    WORK = enum.auto()
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    EXIT = enum.auto()

class Status(enum.Enum):
    PENDING = enum.auto()
    RUNNING = enum.auto()
    SUCCEEDED = enum.auto()
    FAILED = enum.auto()

class SockError(Exception):
    """
    Indicates a socket-related error.
    """

class Sock(socket.socket):
    """
    A socket allowing buffers of arbitrary length as well as complete Python
    objects to be sent and received directly.  See `socket.socket` for more
    information.

    Parameters
    ----------
    token : bytes
        A token used to verify message data.
    out_of_band : bool
        Whether or not to spool message data to temporary files rather than send
        it over the socket directly.  This can be useful if messages are very
        large and filesystem performance exceeds socket performance for a given
        system.
    """

    __slots__ = ("__buf", "__token", "__out_of_band", "__out_of_band_path", "__out_of_band_clean")

    # The format string for message headers.
    __HDR_FMT = "<QQ"
    # The magic number used to ensure a valid message header.
    __HDR_MAGIC = 7091318296757105537 # b"\x81socklib"
    # The length in bytes of a message header.
    __HDR_LEN = struct.calcsize(__HDR_FMT)
    # The maximum length of a block to receive.
    __RECV_LEN = 4096

    # The hash algorithm to use with HMAC to verify message data.
    __HMAC_DIGEST = "sha3_512"
    # The length of the HMAC output.
    __HMAC_LEN = hashlib.new(__HMAC_DIGEST).digest_size

    def __init__(self, *args, token, out_of_band, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a buffer to hold incoming data.
        self.__buf = bytearray()
        self.__token = bytes(token)
        self.__out_of_band = bool(out_of_band)
        self.__out_of_band_clean = []

        if self.__out_of_band:
            with tempfile.NamedTemporaryFile(prefix="multiopenmm_socklib_", suffix=".mmmsock", dir=support.get_scratch_directory(), delete=False) as temp_file:
                pass
            self.__out_of_band_path = temp_file.name
            _debug(f"opened out-of-band communication file {self.__out_of_band_path}")

    def close(self):
        super().close()
        if self.__out_of_band:
            if self.__out_of_band_clean is not None:
                atexit.register(type(self)._remove_out_of_band, self.__out_of_band_path)
                for path in self.__out_of_band_clean:
                    atexit.register(type(self)._remove_out_of_band, path)

    @staticmethod
    def _remove_out_of_band(path):
        _debug(f"removing out-of-band communication file {path}")
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)

    def __repr__(self):
        sockname_host = sockname_port = peername_host = peername_port = "?"
        with contextlib.suppress(OSError):
            sockname_host, sockname_port = self.getsockname()
        with contextlib.suppress(OSError):
            peername_host, peername_port = self.getpeername()
        return f"[{sockname_host}:{sockname_port} -> {peername_host}:{peername_port}]"

    def accept(self):
        """
        Wait for an incoming connection.

        Returns
        -------
        sock : Sock
            A new socket representing the connection.
        addr : object
            The address of the client.  For IP sockets, the address is a pair
            `(hostaddr, port)`.
        """

        # This is the implementation of accept() found in the standard library,
        # except that it creates the same type of client socket as the server
        # socket instead of always creating an instance of socket.socket.
        fd, addr = self._accept()
        sock = type(self)(self.family, self.type, self.proto, fileno=fd, token=self.__token, out_of_band=self.__out_of_band)
        if self.__out_of_band:
            self.__out_of_band_clean.append(sock.__out_of_band_path)
            sock.__out_of_band_clean = None
        if socket.getdefaulttimeout() is None and self.gettimeout():
            sock.setblocking(True)
        return sock, addr

    def send_msg(self, msg):
        """
        Sends a buffer.

        Parameters
        ----------
        msg : buffer
            The bytes to send.
        """
        
        if self.__out_of_band:
            with open(self.__out_of_band_path, "wb") as msg_file:
                msg_file.write(msg)
            msg = self.__out_of_band_path.encode()
        self.sendall(struct.pack(self.__HDR_FMT, self.__HDR_MAGIC, len(msg)) + hmac.digest(self.__token, msg, self.__HMAC_DIGEST) + msg)

    def send_obj(self, obj):
        """
        Sends an object.

        Parameters
        ----------
        obj : object
            The object to send.  It must be picklable (and unpicklable by the
            recipient).
        """

        self.send_msg(pickle.dumps(obj))

    def fill_buf(self):
        """
        Tries to receive as many bytes as possible.

        Returns
        -------
        bool
            Whether or not the sender is still connected.
        """

        try:
            block = self.recv(self.__RECV_LEN)
        except OSError:
            return False
        self.__buf.extend(block)
        return bool(block)

    def recv_msgs(self):
        """
        Tries to process as many buffers as possible from bytes received.
        `fill_buf()` should be called to receive bytes first, followed by
        `recv_msgs()` to process them.

        Yields
        ------
        buffer
            Buffers received.
        """

        while len(self.__buf) >= self.__HDR_LEN + self.__HMAC_LEN:
            # There is at least a message header in the buffer; read the length
            # of the message and see if it is also fully present in the buffer.
            magic, msg_len = struct.unpack(self.__HDR_FMT, self.__buf[:self.__HDR_LEN])
            if magic != self.__HDR_MAGIC:
                raise SockError("invalid message header")
            digest_recv = self.__buf[self.__HDR_LEN:self.__HDR_LEN + self.__HMAC_LEN]
            msg_end = self.__HDR_LEN + self.__HMAC_LEN + msg_len
            if len(self.__buf) < msg_end:
                break

            # If a message is present, verify it, yield it, and then remove it
            # and its length from the buffer to place the next message (if one
            # is present) at the beginning.
            msg = self.__buf[self.__HDR_LEN + self.__HMAC_LEN:msg_end]
            if not hmac.compare_digest(hmac.digest(self.__token, msg, self.__HMAC_DIGEST), digest_recv):
                raise SockError("invalid message digest")
            del self.__buf[:msg_end]

            if self.__out_of_band:
                msg_path = msg.decode()
                with open(msg_path, "rb") as msg_file:
                    msg = msg_file.read()
            yield msg

    def recv_objs(self):
        """
        Tries to process as many objects as possible from bytes received.  See
        `recv_msgs()`.

        Yields
        ------
        object
            Objects received.
        """

        for msg in self.recv_msgs():
            yield pickle.loads(msg)

class _SockWorker:
    __slots__ = ("_sock", "_host", "_port")

    def __init__(self, host, port, token, out_of_band):
        self._sock = Sock(token=token, out_of_band=out_of_band)
        self._connect(host, port)
        self._host, self._port = self._sock.getsockname()

        _log("created", self)

    def __repr__(self):
        return f"<{type(self).__name__}: {self._sock}>"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self._sock.close()

    def run(self, timeout=_DEFAULT_RUN_INTERVAL, count=None):
        """
        Runs a listening loop.

        Parameters
        ----------
        timeout : float
            When specified, wait only this much time, in seconds, listening.
        count : int
            When specified, wait only this many times listening.
        """

        if timeout is None:
            _debug("listening loop started with no timeout")
        else:
            _debug(f"listening loop started with interval {timeout:.3f} s")
        index = -1
        for index in itertools.count() if count is None else range(count):
            sock_list = (self._sock, *self._get_socks())
            r_list, w_list, x_list = select.select(sock_list, (), sock_list, timeout)
            if not self._run(x_list, r_list):
                break
        _debug("listening loop finished at iteration", index)

    def _run(self):
        raise NotImplementedError

    def _get_socks(self):
        raise NotImplementedError

    def _reply(self, sock, *args):
        try:
            sock.send_obj(args)
        except OSError:
            return False
        return True

class Server(_SockWorker):
    """
    An abstract class for a socket server.  Subclasses must implement
    `_get_socks()`, `_handle()`, `_add_client()`, `_drop_client()`, and
    `_continue()`.

    Parameters
    ----------
    host : str, optional
        The hostname.  If none is provided, `socket.gethostname()` will be used.
    port : int, optional
        The port.  If none is provided, an unused port will be used.
    out_of_band : bool, optional
        Whether or not to spool data to temporary files rather than send it over
        sockets directly.
    """

    __slots__ = ("__token", "__out_of_band")

    # The length in bytes of tokens used to verify message data.
    __TOKEN_LEN = 64

    def __init__(self, host=None, port=None, out_of_band=False):
        if host is None:
            host = socket.gethostname()
        if port is None:
            port = 0

        self.__token = secrets.token_bytes(self.__TOKEN_LEN)
        self.__out_of_band = bool(out_of_band)

        super().__init__(host, port, self.__token, self.__out_of_band)

    def _connect(self, host, port):
        self._sock.bind((host, port))
        self._sock.listen()

    @property
    def is_server(self):
        return True

    def write_conn(self, path):
        """
        Writes connection information to a file.  This will occur atomically,
        i.e., the file will never be empty or in a partially written state.

        Parameters
        ----------
        path : str
            The path of the file to write connection information to.
        """
        
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as temp_file:
            pickle.dump((self._host, self._port, self.__token, self.__out_of_band), temp_file)
        os.replace(temp_file.name, path)

    def _run(self, x_list, r_list):
        for sock in x_list:
            if sock is self._sock:
                raise SockError("exceptional condition on server socket")
            else:
                sock.close()
                _log("exceptional condition on client socket", sock)
                if not self._drop_client(sock, True):
                    return False
        for sock in r_list:
            if sock is self._sock:
                client_sock, client_addr = self._sock.accept()
                _log("adding client socket", client_sock)
                if not self._add_client(client_sock, client_addr):
                    return False
            else:
                if sock.fill_buf():
                    for args in sock.recv_objs():
                        if not self._handle(sock, *args):
                            return False
                else:
                    _log("dropping disconnected client socket", sock)
                    if not self._drop_client(sock, False):
                        return False
        return self._continue()
    
    def _handle(self, sock, *args):
        """
        Called when a message is received from a client.  Must return whether or
        not to continue the server loop.
        """

        raise NotImplementedError

    def _add_client(self, sock, addr):
        """
        Called when a new client has connected.  Must return whether or not to
        continue the server loop.
        """

        raise NotImplementedError

    def _drop_client(self, sock, exc):
        """
        Called when a client has disconnected.  Must return whether or not to
        continue the server loop.
        """

        raise NotImplementedError

    def _continue(self):
        """
        Called after processing all waiting clients.  Must return whether or not
        to continue the server loop.
        """

        raise NotImplementedError

class Client(_SockWorker):
    """
    An abstract class for a socket client.  Subclasses must implement
    `_handle()`.

    Parameters
    ----------
    host : str
        The hostname of the server.
    port : int
        The port of the server.
    token : bytes
        A token used to verify message data.
    out_of_band : bool, optional
        Whether or not to spool data to temporary files rather than send it over
        sockets directly.
    """

    __slots__ = ()

    def __init__(self, host, port, token, out_of_band):
        super().__init__(host, port, token, out_of_band)

    def _connect(self, host, port):
        self._sock.connect((host, port))

    @property
    def is_server(self):
        return False

    @classmethod
    def from_conn(cls, path, interval=0):
        """
        Creates a socket client from connection information in a file.  See
        `Server.write_conn()`.

        Parameters
        ----------
        path : str
            The path of the file to read connection information from.
        interval : float, optional
            If a non-zero value is given, repeatedly retry reading from the file
            at this interval in seconds, in case it is initially empty.

        Returns
        -------
        Client
            A socket client.
        """

        # When retrying, we either expect an empty file or a complete pickle.
        # Since Server.write_conn() will do so atomically, this should work.
        while True:
            with open(path, "rb") as conn_file:
                buf = conn_file.read()
                if buf or not interval:
                    break
            time.sleep(interval)
        return cls(*pickle.loads(buf))

    def _run(self, x_list, r_list):
        if x_list:
            raise SockError("exceptional condition on client socket")
        if r_list:
            if not self._sock.fill_buf():
                raise SockError("client lost connection with server")
            for args in self._sock.recv_objs():
                if not self._handle(*args):
                    _log("client exiting normally")
                    return False
        return True

    def _get_socks(self):
        yield from ()

    def _reply(self, *args):
        return super()._reply(self._sock, *args)

    def _handle(self, *args):
        """
        Called when a message is received from the server.  Must return whether
        or not to continue the client loop.
        """

        raise NotImplementedError

class Task:
    __slots__ = ("_server", "_seq")

    def __init__(self, server, seq):
        self._server = server
        self._seq = seq

    def __repr__(self):
        return f"<{type(self).__name__}: {self._seq}>"

    def remove(self):
        self._server.remove_task(self)

    @property
    def status(self):
        return self._server.get_status(self)

    @property
    def ready(self):
        return self.status in (Status.SUCCEEDED, Status.FAILED)

    @property
    def retry(self):
        return self._server.get_retry(self)

    @property
    def data(self):
        return self._server.get_data(self)

    def get_result(self, pop=True):
        return self._server.get_result(self, pop)

_Task = Task
del Task

class QueueServer(Server):
    """
    A socket server handling a task queue.
    """

    __slots__ = ("auto_prune", "__next", "__idle", "__busy", "__tasks",
        "__pending", "__running", "__succeeded", "__failed", "__no_retry")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auto_prune = False

        self.__next = itertools.count()
        self.__idle = {}
        self.__busy = {}
        self.__tasks = {status: {} for status in Status}

        self.__pending = self.__tasks[Status.PENDING]
        self.__running = self.__tasks[Status.RUNNING]
        self.__succeeded = self.__tasks[Status.SUCCEEDED]
        self.__failed = self.__tasks[Status.FAILED]

        self.__no_retry = set()

    def __exit__(self, exc_type, exc_val, traceback):
        for sock in self._get_socks():
            self.__sock_exit(sock)
        super().__exit__(exc_type, exc_val, traceback)

    def run(self, *args, **kwargs):
        if self._continue():
            super().run(*args, **kwargs)
        else:
            _debug("skipping listening loop as no work is waiting")

    def add_task(self, *args, retry=True):
        """
        Adds a task to be run.

        Parameters
        ----------
        *args
            Task inputs.

        Returns
        -------
        Task
            A task object.
        """

        seq = next(self.__next)
        self.__pending[seq] = args
        if not retry:
            self.__no_retry.add(seq)
        return _Task(self, seq)

    def remove_task(self, task):
        """
        Removes a task.

        Parameters
        ----------
        task : Task
            The task to remove.

        Raises
        ------
        ValueError
            The task does not belong to the server or has been removed.
        """

        del self.__tasks[self.get_status(task)][task._seq]
        self.__no_retry.discard(task._seq)

    def get_status(self, task, required=True):
        """
        Checks the status of a task.

        Parameters
        ----------
        task : Task
            The task to check.
        required : bool, optional
            If False, None will be returned if the task has been removed instead
            of raising an exception.

        Returns
        -------
        Status
            The status of the task.

        Raises
        ------
        ValueError
            The task does not belong to the server or has been removed.
        """

        if task._server is not self:
            raise ValueError("task does not belong to server")

        found_in = {status: task._seq in table for status, table in self.__tasks.items()}
        found_count = sum(found_in.values())

        if found_count > 1:
            raise RuntimeError("server internal consistency check failed")
        if found_count:
            return next(status for status, found in found_in.items() if found)
        elif required:
            raise ValueError("task not found")
        return None

    def get_retry(self, task):
        """
        Retrieves whether or not the task can be retried.

        Parameters
        ----------
        task : Task
            The task to check.

        Returns
        -------
        bool
            Whether or not the task can be retried.
        """

        self.get_status(task)
        return task._seq not in self.__no_retry

    def get_data(self, task):
        """
        Retrieves data associated with a task.

        Parameters
        ----------
        task : Task 
            The task to check.

        Returns
        -------
        object
            Input data for pending or running tasks, output data for succeeded
            tasks, and raised exceptions for failed tasks.
        """
        
        return self.__tasks[self.get_status(task)][task._seq]

    def get_result(self, task, pop=True):
        """
        Retrieves the result of executing a task, re-raising any exception
        originally raised by the task.

        Parameters
        ----------
        task : Task
            The task to check.
        pop : bool, optional
            If True, remove the task.

        Returns
        -------
        object
            The task result.

        Raises
        ------
        ValueError
            The task has not completed.
        """

        status = self.get_status(task)
        result = self.get_data(task)
        
        if status not in (Status.SUCCEEDED, Status.FAILED):
            raise ValueError("task incomplete")
        if pop:
            self.remove_task(task)
        if status is Status.FAILED:
            raise result
        return result

    def prune(self):
        """
        Stops unneeded idle clients.
        """

        while len(self.__idle) > len(self.__pending):
            sock = self.__idle.popitem()[0]
            self.__sock_exit(sock)

    def distribute(self, *args, **kwargs):
        """
        Distributes tasks to clients and waits for them to complete.

        Parameters
        ----------
        *args
            Task inputs.
        **kwargs
            Run options.

        Returns
        -------
        list
            Task outputs.
        """
        
        tasks = [self.add_task(*task_args) for task_args in args]
        _debug("distributing", len(tasks), "tasks to any available clients")

        while any(not task.ready for task in tasks):
            self.run(**kwargs)
        return tasks

    def broadcast(self, *args, **kwargs):
        """
        Waits for all clients to become idle, distributes one task to all of
        them, waits for them to complete, and collects the results.

        Parameters
        ----------
        *args
            Task input.
        **kwargs
            Run options.

        Returns
        -------
        list
            Task outputs.
        """

        while self.__busy:
            self.run(**kwargs)

        tasks = []
        while self.__idle:
            task_obj = self.add_task(*args, retry=False)
            seq = task_obj._seq
            task = self.__pending.pop(seq)
            sock = self.__idle.popitem()[0]

            if self._reply(sock, _Kind.WORK, task):
                self.__busy[sock] = seq
                self.__running[seq] = task
                tasks.append(task_obj)
            else:
                sock.close()
                self.__no_retry.remove(seq)

        _debug("broadcasted task to", len(tasks), "clients")
        while any(not task.ready for task in tasks):
            self.run(**kwargs)
        return tasks

    def __sock_exit(self, sock):
        self._reply(sock, _Kind.EXIT)
        sock.close()

    def __debug_all(self):
        _debug(
            len(self.__idle), "idle,",
            len(self.__busy), "busy,",
            len(self.__pending), "pending,",
            len(self.__running), "running,",
            len(self.__succeeded), "succeeded,",
            len(self.__failed), "failed"
        )
    
    def _handle(self, sock, kind, data):
        if kind is _Kind.SUCCESS:
            status = Status.SUCCEEDED
        elif kind is _Kind.FAILURE:
            status = Status.FAILED
        else:
            # Unknown instruction kind from client.
            _log("unknown instruction from client", sock)
            return self._drop_client(sock, False)

        # A client is finished working and is now idle.
        self.__idle[sock] = None
        seq = self.__busy.pop(sock)

        # The task it was working on is now finished.
        del self.__running[seq]
        self.__tasks[status][seq] = data
        return True

    def _add_client(self, sock, addr):
        self.__idle[sock] = None
        return True

    def _drop_client(self, sock, exc):
        sock.close()

        if sock in self.__idle:
            # A client exited or crashed but it was not doing anything.
            del self.__idle[sock]

        else:
            # A client that was working crashed; re-queue its task if desired.
            seq = self.__busy.pop(sock)
            task = self.__running.pop(seq)
            if seq in self.__no_retry:
                self.__no_retry.remove(seq)
                self.__failed[seq] = SockError("lost busy client and no retry requested")
            else:
                self.__pending[seq] = task
        
        return True

    def _continue(self):
        if self.auto_prune:
            self.prune()

        # Give tasks to idle clients.
        while self.__idle and self.__pending:
            seq, task = next(iter(self.__pending.items()))
            sock = self.__idle.popitem()[0]

            if self._reply(sock, _Kind.WORK, task):
                # The client is now actively working on the task we sent.
                self.__busy[sock] = seq
                self.__running[seq] = self.__pending.pop(seq)
            else:
                # Trying to communicate with the client raised an error.  Drop
                # the client (note that it has already been popped from __idle)
                # and leave the task in __pending.
                sock.close()
        
        self.__debug_all()
        
        # Keep running as long as some tasks are not complete.
        return self.__pending or self.__running

    def _get_socks(self):
        yield from self.__idle
        yield from self.__busy

class QueueClient(Client):
    """
    An abstract class for a socket client handling a task queue.  Subclasses
    must implement `work()`.
    """

    __slots__ = ("__state",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__state = {}

    def _handle(self, kind, data=None):
        if kind is _Kind.EXIT:
            # Shut down the client.
            return False

        elif kind is _Kind.WORK:
            # Do work and send back the answer or any exception raised.
            try:
                result = self.work(self.__state, *data)
            except Exception as error:
                _log("client task raised exception")
                traceback.print_exc()
                reply = (_Kind.FAILURE, error)
            else:
                reply = (_Kind.SUCCESS, result)
            if not self._reply(*reply):
                raise SockError("unable to communicate with server")
            return True
        
        else:
            raise SockError("unknown instruction from server")

    def work(self, state, *args):
        raise NotImplementedError

class CallClient(QueueClient):
    """
    A socket client that can make function calls.
    """

    def work(self, state, func, *args, **kwargs):
        return func(state, *args, **kwargs)

@contextlib.contextmanager
def make(path="multiopenmm_socklib.mmmconn", server=QueueServer, client=CallClient, host=None, port=None, out_of_band=False, keep=True, interval=_DEFAULT_CONNECT_INTERVAL):
    """
    Creates a socket server with connection information written to a given path,
    or creates a socket client using the information if the connection file
    already exists.

    Parameters
    ----------
    path : str
        The connection file path.  See `Server.write_conn()`.
    server : type
        The server subclass to instantiate if needed.
    client : type
        The client subclass to instantiate if needed.
    host : str, optional
    port : int, optional
    out_of_band : bool, optional
        Information used to create a socket server if one needs to be created.
        See `Server`.
    keep : bool, optional
        Whether or not to keep the connection file after the server exits.
    interval : float, optional
        The client retry interval.  See `Client.from_conn()`.

    Returns
    -------
    Server or Client
        The socket server or client created.  See `Server.is_server` and
        `Client.is_server`.
    """

    try:
        with open(path, "x") as conn_file:
            pass
    except FileExistsError:
        # The connection file already exists; create a client.
        try:
            with client.from_conn(path, interval) as created_client:
                yield created_client
        except ConnectionRefusedError:
            raise SockError("connection file present but server inaccessible")
    else:
        # The connection file does not exist; create a server.
        try:
            with server(host, port, out_of_band) as server_inst:
                server_inst.write_conn(path)
                yield server_inst
        finally:
            if not keep:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

def serve(*make_args, client_timeout=_DEFAULT_RUN_INTERVAL, client_count=None, **make_kwargs):
    """
    Wraps a function with a `make()` context, either calling the function with a
    server as its first argument or running as a client.

    Parameters
    ----------
    client_timeout : float
        If specified, clients wait only this much time, in seconds, listening.
    client_count : int
        If specified, clients wait only this many times listening.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*wrap_args, **wrap_kwargs):
            with make(*make_args, **make_kwargs) as worker:
                if worker.is_server:
                    return function(worker, *wrap_args, **wrap_kwargs)
                else:
                    worker.run(client_timeout, client_count)
        return wrapper
    return decorator
