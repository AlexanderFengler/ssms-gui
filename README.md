# ssms-gui

## Setting up a Development Environment

If you have Mamba installed, use
```
mamba create -n ssms-gui python=3.11
mamba activate ssms-gui
```

If you have conda installed use

```
conda create -n ssms-gui python=3.11
conda activate ssms-gui
```

Now navigate to the `ssms-gui` folder and install with,

```
pip install ./
```

If you want an editable install use,

```
pip install -e ./
```


## Launching the dashboard
Activate the environment and launch the dashboard script.

``` 
streamlit run src/dashboard-prototype.py
```

## Editing the dashboard 
You can edit the script while the dashboard is live. Pressing `R` while viewing the dashboard will refresh/rerun the script allowing you to view changes without re-launching the dashboard via the command line.

