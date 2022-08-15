#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import getopt
import random
import sys
from decimal import Decimal

# page rank dataset generator
"""
    Usage:
    python dataGen.py -n <number of records> -o <output file>
"""
def main(argv):
    outputfile = ''
    number = 0
    try:
        opts, args = getopt.getopt(argv,"hn:o:",["number=","output="])
    except getopt.GetoptError:
        print('dataGen.py -n <number of records> -o <output file>')
        sys.exit(2)
    if opts.__len__() == 0:
        print('dataGen.py -n <number of records> -o <output file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h' or opt == '':
            print('dataGen.py -n <number of records> -o <output file>')
            sys.exit()
        elif opt in ("-n", "--number"):
            number = int(arg)
        elif opt in ("-o", "--output"):
            outputfile = arg
    if number <= 0:
        print('Number of records must be greater than 0')
        sys.exit(2)
    if outputfile == '':
        print('Output file must be specified')
        sys.exit(2)
    
    print('Output file is "', outputfile)
    print('Number of records is "', number)
    genWordPairsFromWordSet(number, outputfile)

def genWordPairsFromWordSet(number, outputfile):
    # randomly generate a word pair from word set
    initPR = str(Decimal(1)/Decimal(number))
    edge_cnt = 0;
    with open(outputfile, 'w') as f:
        f.write("line to be replaced" + '\n')
        wordSet = genWordSet(number)
        for frompage in wordSet:
            topages = random.sample(wordSet, random.randint(1,min(number, 300)))
            for topage in topages:
                outstr = str(frompage)
                if topage != frompage:
                    outstr += ' ' + str(topage)
                    outstr += '\n'
                    f.write(outstr)
                    edge_cnt += 1
                else:
                    continue

    with open(outputfile, 'r') as f:
         lines = f.readlines()
    lines[0] = str(number) + ' ' + str(edge_cnt) + '\n'
    with open(outputfile, 'w') as f:
        f.writelines(lines)

def genWordSet(number):
    wordSet = set()
    for i in range(number):
        # randomly generate a unique word then insert it into set
        wordSet.add(i)
    return wordSet

if __name__ == "__main__":
    main(sys.argv[1:])
