"""A disposable MySQL server for database-backed tests.

Trimmed from `spyglass/tests/container.py`, which also handles DLC assets,
kachery, and bind-mounted data directories that this project has no use for.
Only constructed when pytest is given `--with-db`.
"""

import time

import datajoint as dj


class DockerMySQLManager:
    """Start, wait on, and tear down a DataJoint MySQL container.

    Parameters
    ----------
    image_name : str, optional
        Docker image, without tag. Default 'datajoint/mysql'.
    mysql_version : str, optional
        Image tag. Default '8.0'.
    container_name : str, optional
        Container name. Default 'mpfc-opto-pytest'.
    port : int, optional
        Host port mapped to the container's 3306. Default 3307, chosen to
        avoid colliding with a developer's own MySQL on 3306.
    password : str, optional
        Root password. Default 'tutorial', matching the DataJoint image.
    verbose : bool, optional
        Print lifecycle messages. Default True.
    """

    def __init__(
        self,
        image_name: str = "datajoint/mysql",
        mysql_version: str = "8.0",
        container_name: str = "mpfc-opto-pytest",
        port: int = 3307,
        password: str = "tutorial",
        verbose: bool = True,
    ):
        import docker  # deferred: only needed under --with-db

        self.client = docker.from_env()
        self.image_name = image_name
        self.mysql_version = mysql_version
        self.container_name = container_name
        self.port = int(port)
        self.password = password
        self.user = "root"
        self.verbose = verbose

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[container] {msg}")

    @property
    def container(self):
        import docker

        try:
            return self.client.containers.get(self.container_name)
        except docker.errors.NotFound:
            return None

    def start(self) -> None:
        """Start the container, reusing an existing one when possible."""
        existing = self.container
        if existing is not None:
            if existing.status != "running":
                existing.start()
                self._say(f"restarted {self.container_name}")
            else:
                self._say(f"reusing running {self.container_name}")
            return

        self.client.containers.run(
            image=f"{self.image_name}:{self.mysql_version}",
            name=self.container_name,
            ports={3306: self.port},
            environment=[f"MYSQL_ROOT_PASSWORD={self.password}"],
            # Synchronous I/O. Slower, more reliable.
            command=["--innodb-use-native-aio=0"],
            detach=True,
        )
        self._say(f"started {self.container_name} on port {self.port}")

    def wait(self, timeout: int = 120, interval: int = 3) -> None:
        """Block until the server accepts a connection, or raise.

        The container reports "running" well before MySQL is ready, so
        readiness is tested by connecting rather than by container status.
        """
        dj.config.update(self.credentials)
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                dj.conn(reset=True)
                self._say("server is accepting connections")
                return
            except Exception as e:  # any failure means not ready yet
                last = e
                time.sleep(interval)
        raise TimeoutError(
            f"MySQL in {self.container_name} not ready after {timeout}s: {last}"
        )

    @property
    def credentials(self) -> dict:
        """DataJoint config for this container."""
        return {
            "database.host": "localhost",
            "database.user": self.user,
            "database.password": self.password,
            "database.port": self.port,
            "safemode": False,
            "custom": {"test_mode": True, "debug_mode": False},
        }

    def stop(self, remove: bool = True) -> None:
        """Stop the container, and remove it unless asked to keep it."""
        container = self.container
        if container is None:
            return
        if container.status == "running":
            container.stop()
        if remove:
            container.remove()
            self._say(f"removed {self.container_name}")
        else:
            self._say(f"stopped {self.container_name}, kept for reuse")
