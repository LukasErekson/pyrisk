#!/usr/bin/env python3
import logging
import os, os.path
import random
import importlib
import re
import collections
from game import Game

from world import CONNECT, MAP, KEY, AREAS
from multiprocessing import Process

import argparse

NAMES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]

def wrapper(stdscr, **kwargs):
    g = Game(screen=stdscr, **kwargs)
    for i, klass in enumerate(kwargs['player_classes']):
        g.add_player(NAMES[i], klass)
    return g.play()

def execute_in_parallel(args, **kwargs):
    process_list = []
    avg_games = args.games // args.processes
    remaining_games = args.games % args.processes
    for i in range(args.processes):
        games = avg_games
        if remaining_games > 0:
            remaining_games -= 1
            games += 1
        p = Process(target=launch_in_process, args=(i, games, args,), kwargs=kwargs)
        print("Starting process {} with {} games".format(i+1,games))
        p.start()
        process_list.append(p)
    for i, p in enumerate(process_list):
        p.join()
        print("Joined process {}".format(i+1))
        
def configure_logger(logger, base_filename, pid, game=None):
    #formatter = logging.Formatter('%(asctime)s - %(name)-14s - %(levelname)-8s - %(message)s')
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    # while logger.hasHandlers():
        # logger.handlers.pop()
    if (base_filename):
        logfile = "logs/{}{}.log".format(base_filename, pid)
        if game is not None:
            logfile = "logs/{}{}.game{}.log".format(base_filename, pid, game)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
   
def launch_in_process(pid, games, args, **kwargs):
    
    wins = collections.defaultdict(int)
    for j in range(games):
        # Use different loggers for each process and agme
        logger = logging.getLogger("pyrisk{}_{}".format(pid, j))
        kwargs['logger'] = logger
        configure_logger(logger, args.log, pid, game=j+1)
        kwargs['round'] = (j+1, args.games)
        kwargs['history'] = wins
        if args.curses:
            import curses
            victor = curses.wrapper(wrapper, **kwargs)
        else:
            victor = wrapper(None, **kwargs)
        wins[victor] += 1

        # Delete the logger objects to save resources.
        logger.handlers[0].close()
        del logger
    #TODO: make below safe for multiprocessing
    print("Process {}: Outcome of {} games".format(pid, games))
    player_classes = kwargs['player_classes']
    for k in sorted(wins, key=lambda x: wins[x]):
        if k == "Stalemate":
            print("%s [%s]:\t%s" % (k, "None", wins[k]))
        else:
            print("%s [%s]:\t%s" % (k, player_classes[NAMES.index(k)].__name__, wins[k]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nocurses", dest="curses", action="store_false", default=True, help="Disable the ncurses map display")
    parser.add_argument("--nocolor", dest="color", action="store_false", default=True, help="Display the map without colors")
    parser.add_argument("-p", "--processes", dest="processes", type=int, default=1, help="The number of processes to run the games in. Note: if p>1 curses will be disabled.")
    parser.add_argument("-l", "--log", nargs="?", action="store", default="", const="pyrisk.log", help="Write game events to a specified logfile")
    parser.add_argument("-d", "--delay", type=float, default=0.1, help="Delay in seconds after each action is displayed")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random number generator seed")
    parser.add_argument("-g", "--games", type=int, default=1, help="Number of rounds to play")
    parser.add_argument("-w", "--wait", action="store_true", default=False, help="Pause and wait for a keypress after each action")
    parser.add_argument("players", nargs="+", help="Names of the AI classes to use. May use 'ExampleAI*3' syntax.")
    parser.add_argument("--deal", action="store_true", default=False, help="Deal territories rather than letting players choose")

    args = parser.parse_args()

    #ensure there is a directory for the log files
    if not os.path.exists("logs/"):
        os.makedirs("logs/")

    if args.log:
        #Note: each process has its own logger so this one isn't used
        logging.basicConfig(filename='logs/{}_other.log', filemode='w')
    elif not args.curses:
        logging.basicConfig()

    if args.seed is not None:
        random.seed(args.seed)

    player_classes = []
    for p in args.players:
        match = re.match(r"(\w+)?(\*\d+)?", p)
        if match:
            #import mechanism
            #we expect a particular filename->classname mapping such that
            #ExampleAI resides in ai/example.py, FooAI in ai/foo.py etc.
            name = match.group(1)
            package = name[:-2].lower()
            if match.group(2):
                count = int(match.group(2)[1:])
            else:
                count = 1
            try:
                klass = getattr(importlib.import_module("ai."+package), name)
                for i in range(count):
                    player_classes.append(klass)
            except:
                print("Unable to import AI %s from ai/%s.py" % (name, package))
                raise
    
    if args.processes > 1:
        args.curses = False
    kwargs = dict(curses=args.curses, color=args.color, delay=args.delay, connect=CONNECT, cmap=MAP,
                  ckey=KEY, areas=AREAS, wait=args.wait, deal=args.deal, player_classes=player_classes)
                  
    if args.games == 1:
        if args.curses:
            import curses
            curses.wrapper(wrapper, **kwargs)
        else:
            wrapper(None, **kwargs)
    else:
        if args.processes > 1:
            execute_in_parallel(args, **kwargs)
        else:
            launch_in_process(0, args.games, args, **kwargs)





