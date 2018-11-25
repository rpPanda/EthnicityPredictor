import os

os.system('googleimagesdownload -k="Chinese Man" -l=100 -f=jpg -o=./train -i=ChineseMan')
os.system('googleimagesdownload -k="American Man" -l=100 -f=jpg -o=./train -i=AmericanMan')
os.system('googleimagesdownload -k="Indian Man" -l=100 -f=jpg -o=./train -i=IndianMan')
os.system('googleimagesdownload -k="African Man" -l=100 -f=jpg -o=./train -i=AfricanMan')
os.system('googleimagesdownload -k="Middle East Man" -l=100 -f=jpg -o=./train -i=ArabMan')
os.system('googleimagesdownload -k="Indonesian Man" -l=100 -f=jpg -o=./train -i=IndonesianMan')

os.system('googleimagesdownload -k="Chinese Woman" -l=100 -f=jpg -o=./train -i=ChineseWoman')
os.system('googleimagesdownload -k="American Woman" -l=100 -f=jpg -o=./train -i=AmericanWoman')
os.system('googleimagesdownload -k="Indian Woman" -l=100 -f=jpg -o=./train -i=IndianWoman')
os.system('googleimagesdownload -k="African Woman" -l=100 -f=jpg -o=./train -i=AfricanWoman')
os.system('googleimagesdownload -k="Middle East Woman" -l=100 -f=jpg -o=./train -i=ArabWoman')
os.system('googleimagesdownload -k="Indonesian Woman" -l=100 -f=jpg -o=./train -i=IndonesianWoman')