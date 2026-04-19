import yt_dlp

url = "https://youtu.be/5YVfuZjbBzE?si=ML3PLReCK5vDK3ja"

ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'video3.mp4'
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])