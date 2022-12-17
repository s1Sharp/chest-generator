import os, time
from threading import Thread


def delete_scheduler(path, maxtime, frequency_time):
    '''
    \:NOTE Thread
    remove oldest .zip files from generated files
    '''
    while True:
        now = time.time()

        for f in os.listdir(path):
            if f.count('.zip') > 0:
                f = os.path.join(path, f)
                if os.stat(f).st_mtime < now - maxtime:
                    if os.path.isfile(f):
                        print(f'removed {f}')
                        os.remove(f)
        time.sleep(frequency_time)


def init_delete_sheduler_job(path, maxtime=24 * 60 * 60, frequency_time=60 * 60):
    thread = Thread(name="DeleteSchedulerThread", target=delete_scheduler, args=(path, maxtime, frequency_time,))
    thread.setDaemon(True)
    thread.start()
