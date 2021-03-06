# CreateInitialModel1D
Create a 1D initial model used for inversion of dispersion curves

The initial model is based on [CRUST 1.0](https://igppweb.ucsd.edu/~gabi/crust1.html)
and [MEAN](http://ds.iris.edu/ds/products/emc-mean/). MEAN reference Earth model is 
based on the Earth model iasp91. MEAN replaces iasp91’s mid-crustal discontinuity and 
Moho depth of 20 and 35 km to 18 and 30 km, respectively. It also replaces iasp91’s a 
high S-velocity zone of the uppermost mantle (to the depth of 210 km) with a low 
S-velocity zone less than 4.5 km/s.

## Examples

```
# first example
python ../create_model.py station.txt --nc 20 --nm 20

# second example
python ../interpolate_model.py mi_raw.txt --zc 90 --zm 200 --dz 2.0 4.0 --smooth 0.7
```
