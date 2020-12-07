# Python one-time setup: from zero to hero

In summary, here is the proposed order to properly set up your work in Python (one-time job).

1. Install Python (e.g. 3.7)
2. Create a *virtual environment* to house all the packages needed in your project
3. Install desired packages in your virtual environment
   * Jupyter (for the notebook experience)
   * Install all desired packages
4. Verify setup, and you're done!



**No Anaconda needed whatsoever!**



### 1. Install Python

To be safe, contact your IT or seek one-on-one help from a colleague to ensure you install Python right.



Here are some things to check off at the end of the installation, to verify.

* Make sure python can be accessed in your terminal (command prompt/bash/etc.) using the command `python`
  * If no luck, then try `python3` instead. This would be an issue if you had multiple versions of Python.
  * If still no luck, then perhaps Python was not added to your system's "PATH".
* For our purposes, we are interested in more recent versions of Python. For example, 3.6 or later might be okay, but I'm not sure. I use 3.7 and it works well.



### 2. Create a virtual environment

It is recommended that for each project you work on, you create a separate virtual env. to better manage your Python packages.



For detailed info on creating and using virtual envs, Corey Schafer's videos are an excellent resource:

* For Windows users ([YouTube](https://www.youtube.com/watch?v=APOPm01BVrk))
* For Mac & Linux users ([YouTube](https://www.youtube.com/watch?v=Kg1Yvry_Ydk))



**Here's a summary for Windows users:**

Say we want to create a virtual environment called `ml_demos`. This will be created as a folder on your machine.

1. Open Command Prompt and change to the desired directory

   * Ideally, this is a directory that's easy to get to (e.g. `...\YOUR_PROJECT\`)
   * To change directories, simply enter `cd` followed by the full path (e.g. `cd C:\...\YOUR_PROJECT\`)
   * Ensure that the path has been updated.

2. Enter `python -m venv ml_demos` 

   * Check that a folder called `ml_demos` has been created in your directory. If so, continue.

3. Activate your virtual environment (i.e. enter into it):

   * `ml_demos\Scripts\activate.bat`
   * You should now see `(ml_demos)` appear on the left

4. Once `ml_demos` is activated (above), enter `pip list` . These two packages should appear:

   ​		`pip`

   ​		`setuptools`

   If a message appears below them indicating that you are using an old version, not to worry. Follow the next step to resolve.

5. Upgrade `pip` and `setuptools`  to ensure smooth package installation later:

   1. `pip install --upgrade pip` ... (refer to Appendix below in case an error appears)
   2. `pip install --upgrade setuptools`

6. Deactivate (i.e. leave) this environment:`deactivate`



**Moving forward, every time you want to activate/deactivate your virtual environment:**

(Again assuming Windows) Open Command Prompt and change to your project directory (like above)

* Activate: `ml_demos\Scripts\activate.bat`
* Deactivate: `deactivate`



### 3. Install desired packages

1. Activate `ml_demos` as described previously.

2. Install Jupyter:

   * `pip install jupyter`
   * `pip install jupyterlab`

3. Install PyTorch (if you need it):

   * Scroll to the "Start Locally" section on [PyTorch's website](https://pytorch.org/get-started/locally/).

     * Select your OS (Linux, Mac, or Windows)
     * For package, **select Pip**
     * For CUDA, this is where it gets tricky. Please refer to Appendix below.

   * Once you selected all of the above, a command should appear in the box below

     next to "Run this Command". **Copy all of this**.

   * Return to your command prompt (or bash, etc.) and paste and run command there.

     * This will take some time (mainly to download ~500 MB)

   * ... and you're done installing PyTorch!

4. Install all remaining packages all at once. In our case, those are all stored in a file called `requirements.txt`.

   * Make sure `requirements.txt` is in the current directory

   * Run this command: `pip install -r requirements.txt`



### 4. Verify setup, and you're done!

Hopefully, you have been able to install everything as described above. Here's one way to test things:

* Testing packages in command-line Python:
  1. Again, make sure `ml_demos` (virtual environment) is activated
  2. Enter Python by simply entering `python`
  3. Try running this command: `import torch, numpy, matplotlib, pandas, sklearn, statsmodels, control, tqdm, gym, Box2D, stable_baselines3`.
  4. Exit python
* Try using JupyterLab:
  * Again, make sure `ml_demos` (virtual environment) is activated
  * Run this command: `jupyter lab`. This should open a new browser window/tab after a few seconds.
  * Have fun with JupyterLab ... (e.g. open notebook, type some Python code, run cells)
    * For more info, YouTube and Google are your best friend!
  * **Once you're done using JupyterLab, it is good practice to end your session as follows:**
    * Shut down any active kernels
      * Click on the stop icon on the left then select which notebooks/kernels to shut down (all)
    * THEN, Shut down entire session: File menu > Shut down
    * You'll now notice, when you go back to your command prompt/bash (which you used to open JupyterLab), that you can type commands again. Before shutting down your session, you are not able to do that.



### APPENDIX

* In step 2, if you encountered an error during `pip install --upgrade pip` and it looked like:

  "ERROR: Could not install packages due to an EnvironmentError: [WinError 5] Access is denied:"

  then you can check if this is actually a problem or not by entering `pip list`. If `pip` appears with no warning message, then you're good to go and you can disregard that error message.

* In step 2, to decide which CUDA version (for installing PyTorch), I recommend you contact your IT.

  * If you're still not sure what option to select, None is a good option (you simply won't be using any GPU).

  * If you are confident you can do it by yourself, here's how to approach it on Windows:

    1. If you don't have GPU on your machine, don't proceed.

    2. If you're using Nvidia, go to Nvidia Control Panel > Help menu > System Information > Components Tab.

       In my case, it showed "NVIDIA CUDA 11.1.70 driver" next to "NVCUDA64.DLL". This indicates that I should select 11.x

       on PyTorch's website.