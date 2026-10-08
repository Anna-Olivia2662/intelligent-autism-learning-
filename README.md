# ABA Alphabet Live Tracer

A desktop app that helps children practise tracing letters A-D.
A teacher video plays, the child traces the letter on a canvas,
and the app scores the tracing and generates an HTML progress report.

## Features
- Teacher demo video for each letter, with replay
- Freehand tracing canvas with a 30-second timer per letter
- Shape scoring using OpenCV (Hu moments, overlap, stroke coverage)
- Guard against dots and random scribbles
- Voice prompts and encouragement (pyttsx3)
- Printable HTML report with scores, attempts and time taken

## Tech stack
Python, Tkinter, OpenCV, NumPy, Pillow, pyttsx3

## How to run
1. Install Python 3.10 or newer
2. `pip install -r requirements.txt`
3. Put your demo videos in a `videos/` folder as A.mp4, B.mp4, C.mp4, D.mp4
4. `python aba_tracer.py`

## How scoring works
The drawing is rendered to a binary image and compared with a template of
the letter. The score blends Hu-moment shape similarity, overlap with the
template, bounding-box spread and stroke amount. 50% or more passes.
This is classical computer vision, not a trained ML model.

## My contribution
This was a team mini project. I worked on the front end: the Tkinter
interface (layout, buttons, progress indicators, feedback messages) and
the drawing canvas, where the child's tracing is drawn as dots that follow
the mouse. The scoring engine was written by my team. I
used AI tools for help while coding and studied my part to understand it.

## Screenshots
![App](screenshots/app.png)
![Report](screenshots/report.png)
