## Package Details
This package is aimed to act as a tuning configurator to find the optimum sharding configs for a model component.
To find the optimum sharding configs, across a variety of variables, define the configurations in 'tuning_configs.yaml'
and trigger the run as follows:

```
cd ${WORKSPACE}/scripts && sh configurator.sh
```

## Features in pipeline:
- Adding pytorch benchmark
- Adding other collectives: 'all_gather', 'reduce_scatter', 'p2p'
- Adding support for tuning specifc blocks, eg GEMMs 
- Adding support for tuning entire layers of given size. eg moe_layer, attention_layer