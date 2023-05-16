<div align="center">

# <b>Mosaic</b>

### Generating The New Yorker Style Cartoons using Stable Diffusion

<img width="350" src="art/cartoon.jpg"/>

</div>

## Introduction

The New Yorker Style cartoons feature an intricate blend of whimsical art style, witty humor, and a subtle commentary on
modern life.  Cartoon enthusiasts and creative professionals alike know that creating such cartoons can be a daunting
task that requires both artistic talent and a knack for satire.

In this project, we aim to explore techniques to simplify the cartoon creation process by using Text-to-Image Diffusion
models to specifically **generate high-quality The New Yorker Style cartoons from natural language captions.**

We demonstrate how to fine-tune a Stable Diffusion model on a custom dataset of {image, caption} pairs. We build on top
of the fine-tuning script provided by Hugging Face [here](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

## Results

**TODO: Complete me**

## Usage

**Note: It's highly recommended that you use a GPU with at least 30GB of memory to execute the code.**

1. Clone the repository
    ```bash
    git clone git@github.com:utsavoza/aperture.git
    ```
2. Setup and activate the virtual environment
    ```bash
    python3 -m venv .
    source ./bin/activate
    ```
3. Install the required dependencies
    ```bash
    pip3 install -r requirements.txt
    ```
4. Configure and execute the fine-tuning procedure
    ```bash
    python main.py
    ```

## References

## License

    Copyright (c) 2023 Utsav Oza

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
