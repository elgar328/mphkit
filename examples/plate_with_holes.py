"""
Heat conduction in a plate with a row of cooling holes.

One end is held at 100 degC; the hole walls are cooled by a coolant at
20 degC (h = 200 W/(m^2*K)), the other end by still air (h = 10 W/(m^2*K)).

The geometry is built with mphkit and the boundaries are selected by
location, so the script works for any number of holes (one or more):
the entity numbers change, the selections do not. Physics, mesh and
study are plain MPh; mphkit reads the heat flowing in.

Usage: python plate_with_holes.py [holes ...]
"""
import sys

import mph

import mphkit as mk


def build_geometry(model, holes):
    """Builds the plate and returns the geometry and its selections."""
    geom = mk.geometry(model, 3, length_unit='mm')
    plate = mk.block(geom, (100, 40, 5), name='plate')
    hole = mk.cylinder(geom, 3, 5, (15, 20, 0), name='hole')
    pitch = 70 / max(holes - 1, 1)
    row = mk.array(geom, hole, size=(holes, 1, 1), displ=(pitch, 0, 0))
    mk.difference(geom, plate, [row])
    model.build(geom)

    selections = {
        'hot end': mk.sel.box(geom, 'boundary', x=0, name='hot end'),
        'cold end': mk.sel.box(geom, 'boundary', x=100, name='cold end'),
        # every hole wall lies in this band; the plate faces reach further
        'hole walls': mk.sel.box(geom, 'boundary', y=(15, 25),
                                 name='hole walls'),
    }
    return geom, selections


def build_model(client, holes):
    """Creates the model with physics, mesh and study."""
    model = client.create(f'plate with {holes} holes')
    geom, selections = build_geometry(model, holes)

    steel = (model/'materials').create('Common', name='steel')
    (steel/'Basic').property('thermalconductivity', ['45[W/(m*K)]'])
    (steel/'Basic').property('density', ['7850[kg/m^3]'])
    (steel/'Basic').property('heatcapacity', ['475[J/(kg*K)]'])

    heat = (model/'physics').create('HeatTransfer', geom, name='heat')
    hot = heat.create('TemperatureBoundary', 2, name='hot end')
    hot.select(selections['hot end'])
    hot.property('T0', '100[degC]')
    cooling = heat.create('HeatFluxBoundary', 2, name='cooling')
    cooling.select(selections['hole walls'])
    cooling.property('HeatFluxType', 'ConvectiveHeatFlux')
    cooling.property('h', '200[W/(m^2*K)]')
    cooling.property('Text', '20[degC]')
    air = heat.create('HeatFluxBoundary', 2, name='cold end')
    air.select(selections['cold end'])
    air.property('HeatFluxType', 'ConvectiveHeatFlux')
    air.property('h', '10[W/(m^2*K)]')
    air.property('Text', '20[degC]')

    (model/'meshes').create(geom, name='mesh')
    study = (model/'studies').create(name='static')
    study.create('Stationary', name='stationary')
    return model, geom, selections


def main(counts):
    client = mph.start()
    for holes in counts:
        model, geom, selections = build_model(client, holes)
        walls = mk.sel.entities(geom, selections['hole walls'])
        print(f'{holes} holes: {len(walls)} hole-wall faces, '
              f'volume {mk.measure(geom, "domain"):.0f} mm^3')
        model.solve()
        temperature = model.evaluate('T', 'degC')
        # ht.ntflux is the flux out of the plate: negative where heat enters
        heat = -mk.integral(geom, 'boundary', 'ht.ntflux',
                            selections['hot end'], unit='W')
        print(f'  temperature from {temperature.min():.1f} to '
              f'{temperature.max():.1f} degC, {heat:.2f} W in at the hot '
              'end')
        client.remove(model)


if __name__ == '__main__':
    main([int(n) for n in sys.argv[1:]] or [2, 5])
