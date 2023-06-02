## Documentations

### Build

First install requirements for building docs:
```
python -m pip install -r docs/requirements.txt
```

Docs can then be built running the commands:

```
cd docs/
sphinx-apidoc -f -o src/ ../torchstain
make html
```

### Usage

To access the documentations, open the generated HTML in your browser, e.g., with firefox on Linux run this:
```
firefox build/html/index.html
```

Alternatively, on macOS you can open the HTML in chrome by:
```
open -a "Google Chrome" build/html/index.html
```

When documentations are pushed to production, they will be available at [torchstain.readthedocs.io/](https://torchstain.readthedocs.io/).
