# LGT-Net - Aroomy
This is an altered implementation of the wonderful paper "[LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network](https://arxiv.org/abs/2203.01824)"(CVPR'22). [[Supplemental Materials](https://drive.google.com/file/d/1vmNoWXdxKc4or2iUKNvkKRTV8pwxSi0J/view?usp=sharing)] [[Video](https://youtu.be/jh0pkaJaOr8)] [[Presentation](https://docs.google.com/presentation/d/1XC3SNCjuXT7m2jjT64UhUA145yBgHJaY/edit?usp=sharing&ouid=116719086346747292409&rtpof=true&sd=true)] [[Poster](https://drive.google.com/file/d/1Uy0qdkDVSARnz4ef9oNgI9tG_UgiuO00/view?usp=sharing)] 

---

**Model Architecture**

Overview of the two main modules to LGT-Net: feature extractor from panorama image, SWG-Transformer blocks.


![network](src/fig/network.png)


---

# Installation

Set a secret API key before starting the service. Create a `.env` file in the project root:

```shell
LGT_SECRET_KEY=your-secret-key-at-least-16-chars
```

For local development only, you can set `LGT_DEV_MODE=true` to use a built-in insecure key instead.

Run the following script to build and start Docker:

```shell
docker compose up --build
```

By default the service is hosted locally at `127.0.0.1:8000`.

# Inference

#### Please see Downloading Pre-trained Weights section first

After the service is running, you can run a health check to see if the model is loaded:

```shell
curl.exe http://127.0.0.1:8000/health
```

`/health` is unauthenticated and returns `200` when healthy or `503` when the model is not loaded.

To predict room layout on a panorama, include the `X-API-KEY` header:

```shell
curl.exe -s -X POST "http://127.0.0.1:8000/predict" `
  -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  -F "image=@path/to/image.jpg" `
  -F "post_processing=manhattan" `
  -F "pre_processing=true" `
  -F "output_3d=false" `
  -o image.json
```

Flags:
- `X-API-KEY`: required API key (must match `LGT_SECRET_KEY`)
- `image`: path to the panorama image (absolute or relative)
- `post_processing`: `manhattan`, `atalanta`, or `original`
- `pre_processing`: pre-draw geometry on the image to improve model performance (`true` by default)
- `output_3d`: generate a 3D mesh and return a download URL (`false` by default)
- `-o`: save the JSON response to the current directory

When `output_3d=true`, the response includes a download URL: `mesh_url` such as `/jobs/{job_id}/mesh`. You can download the mesh with:

```shell
curl.exe -H "X-API-KEY: your-secret-key-at-least-16-chars" `
  "http://127.0.0.1:8000/jobs/{job_id}/mesh" `
  -o room_mesh.obj
```

---

# Downloading Pre-trained Weights

Before the model can be loaded, you must download pre-trained weights and store them exactly as the directory below. Currently this implementation only uses the zind model so you only need to download zind/best.pkl.

- [mp3d/best.pkl](https://drive.google.com/file/d/1o97oAmd-yEP5bQrM0eAWFPLq27FjUDbh/view?usp=sharing): Training on MatterportLayout dataset
- [zind/best.pkl](https://drive.google.com/file/d/1PzBj-dfDfH_vevgSkRe5kczW0GVl_43I/view?usp=sharing): Training on ZInd dataset
- [pano/best.pkl](https://drive.google.com/file/d/1JoeqcPbm_XBPOi6O9GjjWi3_rtyPZS8m/view?usp=sharing): Training on PanoContext(train)+Stanford2D-3D(whole) dataset
- [s2d3d/best.pkl](https://drive.google.com/file/d/1PfJzcxzUsbwwMal7yTkBClIFgn8IdEzI/view?usp=sharing): Training on Stanford2D-3D(train)+PanoContext(whole) dataset
- [ablation_study_full/best.pkl](https://drive.google.com/file/d/1U16TxUkvZlRwJNaJnq9nAUap-BhCVIha/view?usp=sharing): Ablation Study: Ours (full) on MatterportLayout dataset


Make sure the pre-trained weight files are stored as follows:
```
checkpoints
|-- SWG_Transformer_LGT_Net
|   |-- ablation_study_full
|   |   |-- best.pkl
|   |-- mp3d
|   |   |-- best.pkl
|   |-- pano
|   |   |-- best.pkl
|   |-- s2d3d
|   |   |-- best.pkl
|   |-- zind
|       |-- best.pkl
```

# Acknowledgements
The code style is modified based on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

Some components refer to the following projects:

- [HorizonNet](https://github.com/sunset1995/HorizonNet#1-pre-processing-align-camera-rotation-pose)
- [LED2-Net](https://github.com/fuenwang/LED2-Net)
- [PanoPlane360](https://github.com/sunset1995/PanoPlane360)
- [DuLa-Net ](https://github.com/SunDaDenny/DuLa-Net)
- [indoor-layout-evaluation](https://github.com/bertjiazheng/indoor-layout-evaluation)

# Citation
If you use this code for your research, please cite
```
@InProceedings{jiang2022lgt,
    author    = {Jiang, Zhigang and Xiang, Zhongzheng and Xu, Jinhua and Zhao, Ming},
    title     = {LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```
