# A simple rule-based emotion classifier and chatbot
import re

# Simple emotion patterns and responses
emotion_patterns = {
    "joy": (r"\b(glücklich|toll|wunderbar|ausgezeichnet|froh|super|gut|schön)\b",
           r"\b(nicht|kein|keine|keinen)\s+(glücklich|toll|wunderbar|ausgezeichnet|froh|super|gut|schön)\b"),
    "anger": (r"\b(wütend|sauer|frustriert|genervt|verärgert|böse)\b",
             r"\b(nicht|kein|keine|keinen)\s+(wütend|sauer|frustriert|genervt|verärgert|böse)\b"),
    "sadness": (r"\b(traurig|unglücklich|deprimiert|niedergeschlagen|elend)\b",
               r"\b(nicht|kein|keine|keinen)\s+(traurig|unglücklich|deprimiert|niedergeschlagen|elend)\b"),
    "fear": (r"\b(ängstlich|besorgt|nervös|beunruhigt|erschrocken)\b",
            r"\b(nicht|kein|keine|keinen)\s+(ängstlich|besorgt|nervös|beunruhigt|erschrocken)\b"),
    "love": (r"\b(liebe|mag|gefällt|gern|toll)\b",
            r"\b(nicht|kein|keine|keinen)\s+(liebe|mag|gefällt|gern|toll)\b"),
    "surprise": (r"\b(wow|erstaunlich|überrascht|unerwartet|unglaublich)\b",
                r"\b(nicht|kein|keine|keinen)\s+(erstaunlich|überrascht|unerwartet|unglaublich)\b")
}

responses = {
    "joy": "Das freut mich sehr!",
    "joy_negative": "Das tut mir leid zu hören. Möchtest du darüber sprechen?",
    "anger": "Warum bist du verärgert? Ich bin hier, um zu helfen.",
    "anger_negative": "Das ist gut, dass du nicht verärgert bist!",
    "sadness": "Es tut mir leid, das zu hören. Möchtest du darüber sprechen?",
    "sadness_negative": "Das freut mich zu hören!",
    "fear": "Mach dir keine Sorgen. Ich bin bei dir.",
    "fear_negative": "Das ist gut, dass du keine Angst hast!",
    "love": "Das ist schön zu hören.",
    "love_negative": "Oh, das tut mir leid zu hören.",
    "surprise": "Wow, das klingt spannend!",
    "surprise_negative": "Verstehe, das scheint für dich normal zu sein.",
}


def detect_emotion(text):
    text = text.lower()
    for emotion, pattern in emotion_patterns.items():
        if re.search(pattern, text):
            return emotion
    return "neutral"

def chatbot():
    print("Starte einfachen Chatbot. Tippe 'exit' zum Beenden.")
    while True:
        user_input = input("Du: ")
        if user_input.lower() == "exit":
            print("Bot: Bis zum nächsten Mal!")
            break

        emotion = detect_emotion(user_input)
        response = responses.get(emotion, "Interessant! Erzähl mir mehr.")
        print(f"Bot: ({emotion}) {response}")

if __name__ == "__main__":
    chatbot()