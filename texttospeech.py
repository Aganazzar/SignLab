import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

while True:
    text = "what up my friend "
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()
