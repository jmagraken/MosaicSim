# MosaicSim

A program for simulating spatial transformations of the cone photoreceptor mosaic in vertebrate species that undergo a transition from a hexagonal mosaic to a square mosaic.

### Flags:

```--delta```: Maximum distance that a cone may move from its initial position, in single cone diameters.

```--sigma```: The amount by which the double cone domain is expanded radially in each expansion step, in single cone diameters.

```--mu```: The minimum permitted distance between two non-doublable cones after initialization, in single cone diameters.

```--tau```: The minimum permitted initial distance between two single cones chosen to be non-doublable cones, in single cone diameters.

```--initial_mosaic```: The schematic in ```InitialMosaics/``` from which to initialize the mosaic. Can be any of AH1, AH2, AH3, AH4, AH5, Sab1, Sab2, Sab3, Sab4, Sab5.

```--out_dir```: Path of a directory to save the output.

```--verbose```: Prints simulation progress to the console.
