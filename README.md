# FFReID
This is the offical implementation of the published papers 'Improving Federated Person Re-Identification through Feature-Aware Proximity and Aggregation'.

P. Zhang, H. Yan, W. Wu, and S. Wang, “Improving Federated Person Re-Identification through Feature-Aware Proximity and Aggregation,” ACM International Conference on Multimedia (MM), Ottawa, CAnada, Oct. 2023.

## Run


Run the experiments for the test. You just need to replace the ImageNet pre-trained model in the test folder with our trained model.
```
python test.py
```
## Our trained model: [Fully supervised model](https://drive.google.com/file/d/1p-2w4JED3VgSTDFpcUsERagX5Oqq2uSn/view?usp=drive_link). In the domain generalization experiment setting, our trained model is as follows: [Market](https://drive.google.com/file/d/1R9YK4AhCuVPK8Pzw5HIqaeagkkmOZ7XX/view?usp=drive_link), [DukeMTMC](https://drive.google.com/file/d/16bpt3j8bZAwNTTLz72zQSbC7MiOAKR3Y/view?usp=drive_link), [MSMT17](https://drive.google.com/file/d/1rypfaisYbX_z4GKUvbQPdSsBhn5Xm6qf/view?usp=drive_link)


## Please refer to the [FedPaV](https://github.com/cap-ntu/FedReID) for the implementation of the data processing files, as well as the FedAvg and FedPaV methods:
