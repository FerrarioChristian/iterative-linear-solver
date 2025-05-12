#!/usr/bin/bash

echo "Argomento passato: $1"

if [ ${1,,} = Andrea ]; then
	echo "ok"
elif [ ${1,,} = help ]; then
	echo "halp"
else
	echo "wut"
fi
