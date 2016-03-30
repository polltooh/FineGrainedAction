from pytube import YouTube
import string
import os

def download_single_video(label, video_url):

    yt = YouTube(video_url)
    try:
        video = min(yt.filter('mp4'))
    except:
        video = min(yt.get_videos())
     
    save_dir = "video/" + label + "/"
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    valid_chars = "-_ %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in yt.filename if c in valid_chars)

    filename = filename.replace('-','_')
    filename = filename.replace(' ','_')
    while(filename[-1] == '_'):
        filename = filename[:-1]

    if(os.path.isfile(save_dir + filename + '.' + video.extension)):
        return

    yt.set_filename(filename)
    video.download(save_dir)


if __name__ == "__main__":
    download_single_video("nba_dunk", "https://www.youtube.com/watch?v=A2IMbbRdlsI")
