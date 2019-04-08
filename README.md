# Automatically detecting buildings from satellite images.
aws s3 cp  s3://spacenet-dataset/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz spacenet-dataset/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz

### Related Links:
- [Topcoder](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16892&pm=14551)

### Dataset

For now we are exploring the **spacenet-dataset/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz**

```bash
aws s3 cp s3://spacenet-dataset/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz spacenet-dataset/AOI_2_Vegas/AOI_2_Vegas_Train.tar.gz
```

Exploring the data available in the sample data_set:
1. Open and visualize the 4 types of rasters (PAN, MUL, RGB_PanSharpen and MUL_PanSharpen);
2. Read metadata information;
3. Open the geojson files and overlap them on a raster;
