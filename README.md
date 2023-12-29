# Total-perspective-Vortex
This subject aims to create a brain computer interface based on electroencephalographic data (EEG data) with the help of machine learning algorithms. Using a subject’s EEG reading, you’ll have to infer what he or she is thinking about or doing - (motion) A or B in a t0 to tn timeframe.



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

