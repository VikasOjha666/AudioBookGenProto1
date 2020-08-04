import argparse
from pdf2image import convert_from_path, convert_from_bytes
import os


parser=argparse.ArgumentParser()
parser.add_argument('--pdfname')
parser.add_argument('--usegtts',default='No')
args=parser.parse_args()

if args.usegtts=='No':
	from RecognizePage_rescale import generate_audios
else:
	from RecognizePagegTTS import generate_audios






images = convert_from_path(f'./{args.pdfname}.pdf')

i=0
for img in images:
	img.save(f'./pageimgs/{i}.jpg')
	i+=1

pagelist=os.listdir('./pageimgs/')
pagelist.sort(key=lambda x:int(x.split(".")[0]))

for page in pagelist:
	generate_audios(args.pdfname,'./pageimgs/'+page)
	os.remove('./pageimgs/'+page)
	


