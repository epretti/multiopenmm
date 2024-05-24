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

import abc
import itertools

from . import simulation
from . import support

class Manager(abc.ABC):
    """
    A generic task manager supporting concurrent execution of OpenMM
    simulation-related tasks.

    Parameters
    ----------
    platform_data : multiopenmm.PlatformData, optional
        If provided, custom platform information to be used at context creation.
    """

    __slots__ = ("__platform_data", "__token_counter")

    def __init__(self, platform_data=None):
        self.platform_data = platform_data
        self.__token_counter = itertools.count()

    @property
    def platform_data(self):
        """
        multiopenmm.PlatformData: Custom platform information to be used at
        context creation.

        If ``None``, OpenMM will be permitted to choose a default platform and
        platform parameters when creating contexts.
        """

        return self.__platform_data

    @platform_data.setter
    def platform_data(self, platform_data):
        if platform_data is not None and not isinstance(platform_data, simulation.PlatformData):
            raise TypeError("platform_data must be a simulation.PlatformData or None")

        self.__platform_data = platform_data

    def _get_token(self):
        # Retrieves a unique (for this manager) integer that can be used to
        # unambiguously identify a task.

        return next(self.__token_counter)

    @abc.abstractmethod
    def _distribute(self, *requests):
        # Should return a collection of callable response objects for each
        # callable request object that, when called, block until the
        # corresponding request has finished executing, and either return its
        # result or raise an exception raised by the request.

        raise NotImplementedError

    @abc.abstractmethod
    def _get_response(self, token):
        # Should retrieve response data identified by a token previously issued
        # as a response by the manager.

        raise NotImplementedError

class Response:
    # Represents a response returned by a manager.

    __slots__ = ("__manager", "__token")

    def __init__(self, manager, token):
        if not isinstance(manager, Manager):
            raise TypeError("manager must be a Manager")
        if not isinstance(token, int):
            raise TypeError("token must be an int")

        self.__manager = manager
        self.__token = int(token)

    def __call__(self):
        return self.__manager._get_response(self.__token)

class SynchronousManager(Manager):
    """
    A task manager supporting execution of a single task at a time.  All tasks
    are executed synchronously and serially upon submission, and results are
    made available immediately after completion of all tasks.
    """

    __slots__ = ("__client", "__results")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__client = simulation.Client()
        self.__results = {}

    def _distribute(self, *requests):
        responses = []

        for request in requests:
            request_with_platform_data = support.Arguments(*request.args, **request.kwargs, platform_data=self.platform_data)
            try:
                data = request_with_platform_data.apply_to(self.__client.execute)
                success = True
            except Exception as exception:
                data = exception
                success = False

            token = self._get_token()
            self.__results[token] = (success, data)
            responses.append(Response(self, token))

        return tuple(responses)
    
    def _get_response(self, token):
        try:
            success, data = self.__results.pop(token)
        except KeyError:
            raise ValueError("response already evaluated")

        if success:
            return data
        else:
            raise data

class SocketServerManager(Manager):
    """
    A task manager supporting execution of multiple tasks through a socket
    server.  Any number of clients can connect to the server, and tasks will be
    distributed among the available clients.  Clients may be connected while the
    server is running and other clients are processing tasks; new tasks will be
    distributed to the new clients as well as the existing ones.  Clients may be
    disconnected while they are idle or busy; interrupted tasks will be
    redispatched to other clients.  Results are made available as they are
    returned from clients.
    """

    raise NotImplementedError
