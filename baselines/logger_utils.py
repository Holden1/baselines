from datetime import datetime
from os import path as osp

def path_with_date(path, game):
    """
    :param path: Path to file/folder
    :param game: Name of game e.g. "Pong"
    :return: String consisting of path + date and name of game.
    """
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return osp.join(path, osp.join(game, date))

def log_call_parameters(dir, params):
    """ Logs the call parameters (params) as a text file in dir """
    with open(dir + "/parameters.txt", "w") as text_file:
        text_file.write(str(params))

if __name__ == "__main__":
    print("Utility module. Nothing to run here.")