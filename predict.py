import torch
import yaml
import fastvqa.models as models
import fastvqa.datasets as datasets
import numpy as np

def predict_video_quality(video_path, model_path, opt_path):
    # Load options from YAML file
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    state_dict = torch.load(model_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Prepare the dataset
    # Modify the dataset options to include only the single video
    dataset_opts = opt["data"]["val"]["args"]
    dataset_opts["root"] = [video_path]  # Assuming the dataset can accept a list of video paths
    dataset_opts["name"] = ["test_video"]  # Assign a name to the video
    dataset_opts["label"] = [0]  # Dummy label

    # Create an instance of the dataset for a single video
    video_dataset = getattr(datasets, opt["data"]["val"]["type"])(dataset_opts)

    # Since we have only one video, we can get the data directly
    data = video_dataset[0]

    # Prepare video input
    video = {}
    sample_types = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            b, c, t, h, w = video[key].unsqueeze(0).shape  # Add batch dimension
            video[key] = video[key].reshape(
                b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
            )
            video[key] = video[key].permute(0, 2, 1, 3, 4, 5)
            video[key] = video[key].reshape(
                b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
            )

    # Pass through the model
    with torch.no_grad():
        labels = model(video, reduce_scores=False)
        # Get the mean value of the predictions
        labels = [np.mean(l.cpu().numpy()) for l in labels]

    # Return the predicted quality score
    return labels[0]  # Assuming a single label is returned

# Example usage:
video_quality_score = predict_video_quality('path_to_video.mp4', 'model_checkpoint.pth', 'options.yaml')
print(f"Predicted Video Quality Score: {video_quality_score}")
