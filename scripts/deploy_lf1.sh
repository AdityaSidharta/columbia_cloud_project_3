##!/bin/bash

rm -f lf1.zip
rm -f function/lambda_function.py
cp lambda/lf1.py function/
mv function/lf1.py function/lambda_function.py
cd function
zip ../lf1.zip lambda_function.py
cd ../package
zip -r ../lf1.zip .
cd ../
aws s3 cp lf1.zip s3://spamemail-function/lf1.zip
rm -f function/lambda_function.py
touch function/lambda_function.py