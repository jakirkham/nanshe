import errno
import shutil


def rmtree(*args, **kwargs):
    n = 4
    for i in range(n):
        try:
            shutil.rmtree(*args, **kwargs)
            break
        except OSError as e:
            if e.errno == errno.ENOENT:
                break
            elif i == (n - 1):
                raise
