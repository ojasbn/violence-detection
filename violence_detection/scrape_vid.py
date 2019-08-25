from bs4 import BeautifulSoup as bs
import pytube
import requests


base = "https://www.youtube.com/results?search_query="

#Alter query string for different video
qstring = "boddingtons+advert"


r = requests.get(base+qstring)

page = r.text
soup = bs(page,'html.parser')

vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})

videolist=[]
for v in vids:
    tmp = 'https://www.youtube.com' + v['href']
    videolist.append(tmp)


count=0
for item in videolist:
 
    # increment counter:
    try:
    	count+=1
    	file_name = 'Video_'+str(count)
 
    # initiate the class:
    	pytube.YouTube('https://youtu.be/9bZkp7q19f0').streams.first().download(
  		output_path='./', 
  		filename=file_name,
	)
    except :
    	pass
    
 
    # have a look at the different formats available:
    #formats = yt.get_videos()
 
    # grab the video:
    
 
    # set the output file name:

    
 
    # download the video:
    



