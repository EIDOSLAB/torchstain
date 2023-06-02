## Documentations

### Build

Docs can be built by running the commands:

```
cd docs/
python -m sphinx -T -E -b html -d _build/doctrees -D language=en . html
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
