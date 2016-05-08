#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <gold-file> <test-file>"
  exit 1
fi

EVALB=/afs/crc.nd.edu/group/nlp/data/treebank/bin/EVALB

$EVALB/evalb -p $EVALB/unlabeled.prm $2 $1
