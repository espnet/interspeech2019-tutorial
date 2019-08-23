# INTERSPEECH2019 Tutorial materials

## Setup

Install jupyter & RISE.

```bash
pip install -U jupyter
pip install -U matplotlib
pip install -U RISE
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix
```

Copy `custom` directory including CSS for jupyter notebook.

```bash
cp -r custom ~/.jupyter
```

Update configuration of RISE.

```bash
python config_updater.py
```

## How-to-apply CSS for RISE

Make `<notebook_name>.css` and then put it in the same directory of the notebook.  
The CSS will be automatically loaded in presentation mode.

## Which css part is correspoing to the slide?

1. Start presentation with chrome
2. Right click the part which you want to know -> 検証 -> Happy!

## How-to-export as PDF with CSS

See https://www.procrasist.com/entry/5-jupyter-slide
