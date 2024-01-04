Here is an example of calculating heat diffusion by using the graph approach. 

In the repository are gifs that display the heat diffusion process in 4 example cases. They are displayed as grids for the simplicity of visualization.

To create your own example, create a black and white image with the size 100x100 pixels. The black pixels will not be included in the graph, meaning that the temperature can not diffuse through them. Add the image to the same folder and edit the image name in parameters.py. Then edit parameters in parameters.py however you desire and run heat_diffusion_simpler.py and a gif will be saved to display the process.


Here are some examples:

Coffee in a perfect cup: 

![](./gifs/coffee.gif)

Initial coffee temperature is 90 and environment temperature is 1.



Office heater:

![](./gifs/office.gif)

Heater has a constant temperature of 30 and environment is 1.


Candle:

![](./gifs/candle.gif)

Candle fire has a constant temperature of 500 and environment is 1.


Abstract art:

![](./gifs/abstract.gif)

Initial temperature is 30 and environment is 1.
