import os
import subprocess
import argparse


parser = argparse.ArgumentParser(description='Give the numbers of the file you want to treat')
parser.add_argument('--m', help = "Put the month to analyze, ex : 1920/m/01", default = '10')
parser.add_argument('--fd', help = "Put the first day, ex : 1920/10/n1", default = '01')
parser.add_argument('--ld', help = "Put the last day, ex : 1920/10/n2", default = '31')
args = parser.parse_args()
for i in range(int(args.fd),int(args.ld)+1):
    subprocess.Popen("python ocr.py --path 1920/" + args.m +'/'+ "{:02d}".format(i), creationflags=subprocess.CREATE_NEW_CONSOLE)



