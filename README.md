
<div style="text-align: center;">
<img src="https://i.ytimg.com/vi/1vepY9ekyL8/mqdefault.jpg" width="500px">
</div>

# Total-perspective-Vortex
This subject aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG reading, we infer what he or she is thinking about or doing - (motion) A or B in a `t0` to `tn` timeframe.


## Introduction Story made by Bard (Google AI)

In the bustling city of Neuronsville, nestled deep within the skull, resides a hidden language, whispered in electrical waves. This is the secret language of thoughts, emotions, and dreams – the language of EEG. Ever wondered how a single thought can light up a room, or a catchy tune can set your feet tapping? This symphony of electricity, conducted by billions of tiny neurons, is the maestro behind it all.

But deciphering this language isn't as easy as eavesdropping on a conversation. Enter the .edf file – a digital Rosetta Stone for brain waves. Imagine it as a map, painstakingly drawn by scientists, guiding us through the labyrinthine streets of Neuronsville. Each squiggle and dip on this map represents a tiny electrical pulse, whispering tales of what's happening inside your head.

Now, picture yourself as a detective, armed with the latest MNE technology (your magnifying glass and decoder ring in this analogy). Your mission: to crack the code of your own brainwaves in an .edf file! Follow the trail of these electrical dances, filtering out the distracting city noise of muscle twitches and eye blinks. Focus on the rhythmic hum of alpha waves, the playful flickers of beta, and the slow, deliberate thrum of theta. Each band tells a different story – alpha, the relaxed storyteller, beta, the energetic party animal, and theta, the wise old sage reminiscing about dreams.

But the real fun begins when you start extracting features – the juicy bits of information hidden within the waves. Imagine them as hidden treasure chests, bursting with clues about your mental state. Are you focused and attentive? The beta band might hold the key. Feeling sleepy? Theta waves could be your telltale sign. Did you just have a brilliant idea? Look for a sudden spike in alpha!

With each feature you extract, you piece together the puzzle of your mind. You can train smart assistants to respond to your brainwaves, control virtual worlds with your thoughts, or even diagnose conditions like epilepsy by reading the unique patterns in your electrical chatter.

So, the next time you close your eyes and let your imagination soar, remember – a hidden symphony is playing within you, waiting to be deciphered. Dive into the world of EEG and .edf files, become a detective of your own mind, and unlock the secrets of your brain's electric language!

<img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW1lMm1kMWtqdTY4NDYyeDRkdWg3MWx4bnlvc3Nkb251dDgycmZyNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ohzdZ93WjdUo0dGdq/giphy.gif" width="80%"/>

## What is EEG ?


Imagine a universe within your skull, buzzing with electrical activity. This vibrant tapestry, woven by billions of neurons firing, holds the secrets of your thoughts, emotions, and consciousness. Electroencephalography (EEG) lets us glimpse into this hidden world, offering a window into the electrical language of your brain.

Think of EEG as a microphone for your neurons. Instead of capturing sound waves, it picks up minute fluctuations in voltage generated by your brain cells. These electrical whispers are then translated into squiggles and dips on an .edf file, a kind of map of your brain's electrical landscape.

But reading this map requires some detective work. The raw EEG signal is cluttered with interference – electrical noise from muscles, eyes, and even the environment. Enter MNE, a sophisticated tool akin to a noise-cancelling headset for your brainwaves. It helps filter out the distractions, leaving only the pure language of your neurons.

Once the noise is cleared, the fun begins – feature extraction. This is like picking out the key words from a conversation, gleaning essential information from the electrical patterns. We can analyze the power of different brainwave bands, each telling its own story:

* **Alpha waves:** The chilled-out storyteller, associated with relaxation and creativity.
* **Beta waves:** The energetic party animal, buzzing when we're focused and alert.
* **Theta waves:** The wise old sage, whispering during sleep and daydreams.

By studying these features, we unlock a treasure trove of insights:

* **Emotional states:** A surge in beta may signal stress, while theta waves could paint a picture of drowsiness.
* **Cognitive processes:** Focused attention dances with beta rhythms, while theta waves might guide imaginative journeys.
* **Brain health:** Unique patterns in EEG data can even hold clues to neurological conditions like epilepsy.

EEG is more than just a scientific marvel; it's a bridge between our inner world and the external world. Imagine controlling robots with your thoughts, designing therapies tailored to individual brain patterns, or even unlocking the secrets of consciousness.

So, the next time you close your eyes, remember – a symphony of electricity plays within you, waiting to be heard. Grab your metaphorical magnifying glass and dive into the fascinating world of EEG. Who knows what hidden wonders your brain may whisper?


![Context Image about EEG](https://www.researchgate.net/profile/Sebastian-Nagel-4/publication/338423585/figure/fig1/AS:844668573073409@1578396089381/Sketch-of-how-to-record-an-Electroencephalogram-An-EEG-allows-measuring-the-electrical.png)
### setup enviroment:
- Install MNE first, follow the official wesite [here](https://mne.tools/stable/install/index.html).
    
- Create Virtual Environment and Activate
    - `python3 -m venv tenv`
    - `source tenv/bin/activate`

- Install System Packages (if not already installed)
    - `sudo apt-get install libblas-dev liblapack-dev libffi-dev libgfortran5`

-  Install Python Packages including Pillow
    - `pip install matplotlib mne ipykernel Pillow`
-  enable widgets extensions in jupyter:
    - `jupyter nbextension enable --py widgetsnbextension`

-  Create Jupyter Kernel for Virtual Environment
    - `python -m ipykernel install --user --name tenv --display-name "Python3 (tenv)" `

-  Start JupyterLab (make sure to run it inside the project folder)
    - `jupyter lab`

-  Select Virtual Environment Kernel in Jupyter Interface
    - Open Jupyter Notebook or JupyterLab in a web browser.
    - open desired *.ipynb
    - In the "`Kernel`" menu, select "`Change kernel`" and choose the kernel named "`Python3 (tenv)`" (or the name you specified).


### Ressources:
- [Automated Classification of L/R Hand Movement EEG Signals using Advanced Feature Extraction and Machine Learning, `Mohammad H. Alomari, Aya Samaha, and Khaled AlKamha`,  No. 6, 2013](https://arxiv.org/pdf/1312.2877.pdf)