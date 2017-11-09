import win32ui


def window_exists(window_name):
    try:
        win32ui.FindWindow(None, window_name)
        return True
    except win32ui.error:
        return False

print(window_exists("Error"))