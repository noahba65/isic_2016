import argparse
import os
import torch
from torch import nn
from poutyne import Model, CSVLogger
from poutyne.framework import ModelCheckpoint, EarlyStopping, plot_history
import numpy as np
import torchmetrics
from datetime import datetime
import sys
import pandas as pd
from custom_lib.custom_models.basic_nn import NeuralNetwork
from custom_lib.data_prep import data_transformation_pipeline, SkinLesionDataset
import matplotlib as plt
import torchvision.models as models
import time
import importlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from poutyne import ReduceLROnPlateau, Callback
from thop import profile
from custom_lib.eval_tools import metrics_generator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def plot_cm(y_true, y_pred, results_dir, fname):
    # Convert predictions to class labels if necessary
    y_pred_to_class = np.argmax(y_pred, axis=1)  # Assuming y_pred is the output from the model
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred_to_class)

    # Create the heatmap plot with custom labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.savefig(f"{results_dir}/{fname}.png", bbox_inches='tight')
    plt.close()  # Closes the current figure




def load_model(model_name, **kwargs):
    """Dynamically loads and instantiates a model from custom_lib.custom_models."""
    module = importlib.import_module(f"custom_lib.custom_models.{model_name}")
    
    # Find the first class in the module (assuming only one model class per file)
    model_class = getattr(module, model_name, None)
    
    if model_class is None:
        raise ValueError(f"Could not find a class named '{model_name}' in '{module.__name__}'")

    return model_class(**kwargs)


class PrintLRSchedulerCallback(Callback):
    def set_model(self, model):
        self.model = model  # Store the model reference

    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Current LR = {lr}")

def compute_model_stats(model, batch_size, image_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Computes the FLOPs and number of parameters for a given model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        batch_size (int): The batch size for the dummy input.
        image_size (int): The height and width of the input image.
        device (str): The device to perform computations on ("cuda" or "cpu").
    
    Returns:
        tuple: (GFLOPs, parameters)
    """
    model.to(device)  # Move model to the specified device
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)

    flops, params = profile(model, inputs=(dummy_input,))
    gflops = flops / 1e9  # Convert to GFLOPs

    print(f"GFLOPs: {gflops:.3f}")
    print(f"Parameters: {params:,}")  # Add commas for readability

    return gflops, params       

def main(args):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    print(f"Using {device} device")

    ############################# Loading and Transforming Data #############################################

    # Set data transform parameters
    train_transform = data_transformation_pipeline(image_size = args.image_size,
                                                   center_crop=args.center_crop,
                                                   rotate_angle=args.rotate_angle,
                                                   horizontal_flip_prob=args.horizontal_flip_prob,
                                                   gaussian_blur=args.gaussian_blur,
                                                   normalize=args.normalize,
                                                   brightness_contrast_range=args.brightness_contrast_range,
                                                   is_train=True)
    
    test_transform = data_transformation_pipeline(image_size = args.image_size,
                                                 center_crop=args.center_crop,     
                                                 normalize=args.normalize,   
                                                 rotate_angle = None,
                                                 horizontal_flip_prob = None,
                                                 gaussian_blur = None,
                                                 brightness_contrast_range = None,
                                                 is_train=False)
    
    # Data path
    train_path = f"{args.data_dir}/{args.data_folder}/train"
    test_path = f"{args.data_dir}/{args.data_folder}/test"

    train_labels_path = f"{train_path}/labels.csv"
    test_labels_path = f"{test_path}/labels.csv"

    train_image_path = f"{train_path}/images"
    test_image_path = f"{test_path}/images"

    # num_classes = len(data.classes)

    train_data = SkinLesionDataset(
        csv_file=train_labels_path,
        img_dir=train_image_path,
        transform=train_transform
    )

    test_data = SkinLesionDataset(
        csv_file=test_labels_path,
        img_dir=test_image_path,
        transform=test_transform
    )

    # Split or use directly
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    test_loader = DataLoader(test_data, batch_size=32 * 2, shuffle=False)
    
    num_classes = train_data.data['label'].nunique()

    ############################################################################################
    ################ Define Model and set Callbacks ####################
    model = load_model(
                args.model_name,
                num_classes=num_classes,
                removed_layers=args.truncated_layers,
                batch_size=args.batch_size,
                image_size=args.image_size,
                pretrained=args.pretrained,
                dropout_p=args.dropout_p
                        )
    
    if args.save_logs:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Create directory for saving all logs and model outputs 
        results_dir = os.path.join(f"{args.results_folder_name}/{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Logs and output will be saved in: {results_dir}")
    
    # num_pos = 2495
    # num_neg = 514
    # pos_weight = torch.tensor([num_neg / num_pos])  # This adjusts the loss for imbalance
        
    poutyne_model = Model(
                        model,
                        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
                        loss_function=nn.CrossEntropyLoss(),
                        batch_metrics=["accuracy"],
                        device=device
                        )
    
    if args.save_logs == True:
        # Callback: Save the best model based on validation accuracy
        checkpoint = ModelCheckpoint(f"{results_dir}/best_model.pth", monitor='loss', mode='min', save_best_only=True)
        csv_logger = CSVLogger(f"{results_dir}/training_logs.csv")
        callbacks = [checkpoint, csv_logger]
        
    ############################################################################################
    ################ Train Model ####################

    start_time = time.time()
    # 7. Train the model
    history = poutyne_model.fit_generator(train_loader, epochs=args.epochs, verbose=True,
                                callbacks = callbacks)
    end_time = time.time()

    run_time = end_time - start_time

    print(f"Model training took {run_time / 60} minutes")

    # Save the final model if save_logs is True 
    if args.save_logs:
        torch.save(poutyne_model.network.state_dict(), f"{results_dir}/final_model.pth")

    # If save_logs is True, load the best peforming model by min validation loss for evaluation
    if args.save_logs:
        best_model_path = f"{results_dir}/best_model.pth"
        
        # Load the state dict into the model
        poutyne_model.network.load_state_dict(torch.load(best_model_path))


    ############################################################################################
    ################ Testing model on internal test data ####################
    print("Starting test evalution")

    # Evaluate using Poutyne
    test_loss, test_acc, y_pred, y_true = poutyne_model.evaluate_generator(test_loader, 
                                                                           return_pred=True, 
                                                                           return_ground_truth=True)

    print("Starting specificity and sensitivity evaluation")
    

    sen, spec = metrics_generator(y_pred=y_pred, y_true=y_true)

    print("Sensitivity: ", sen)
    print("Specificity: ", spec)

    # Plot and save confusion matrix
    if args.save_logs:
        plot_cm(y_true=y_true, y_pred=y_pred, 
                   results_dir=results_dir, fname = "confusion_matrix")



    ##########################################################################################

    # Calculate and print Giga FLOPS and model parameters
    gflops, params = compute_model_stats(model, batch_size=1, image_size=args.image_size)
  
    ######################### Saving results ###########################################

    # Save logs and plots
    if args.save_logs:
        with open(f"{results_dir}/model_overview.txt", "w") as file:
            file.write(f"Model Structure:\n{model}\n")
            file.write(f"Using {device} device\n")

        # Check if CSV exists
        if os.path.exists(f"{args.results_folder_name}/test_results.csv"):
            test_results_df = pd.read_csv(f"{args.results_folder_name}/test_results.csv")
        else:
            test_results_df = pd.DataFrame()



        # Create a DataFrame for the new model's metadata
        new_results_df = pd.DataFrame({
            "model_id": [f"{args.model_name}_reduced_layers_{args.truncated_layers}_{timestamp}"],
            "model": [args.model_name],
            "truncated_layers": [args.truncated_layers],
            "epochs": [args.epochs],  
            "batch_size": [args.batch_size],
            "run_time": [run_time / 60],  
            "lr": [args.lr],
            "image_size": [args.image_size],  
            "rotate_angle": [args.rotate_angle],  
            "horizontal_flip_prob": [args.horizontal_flip_prob],  
            "gaussian_blur": [args.gaussian_blur],  
            "brightness_contrast_range": [args.brightness_contrast_range],
            "normalize": [args.normalize],
            "seed": [args.seed],
            "gflops": [gflops],
            "params": [params],
            "test_acc": [test_acc],
            "test_loss": [test_loss],
            "sensitivity": [sen],
            "specificity": [spec],
            })


        # Append to existing DataFrame
        test_results_df = pd.concat([test_results_df, new_results_df], ignore_index=True)

        # Save updated results
        test_results_df.to_csv(f"{args.results_folder_name}/test_results.csv", index=False)

        # Plot training history
        plot_history(
            history,
            metrics=['loss', 'acc'],
            labels=['Loss', 'Accuracy'],
            titles=f"{args.model_name} Training",
            save=True,  
            save_filename_template='{metric}_plot',  
            save_directory=results_dir,  
            save_extensions=('png',)  
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a truncated EfficientNet model.")
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset.")
    parser.add_argument("--data_folder", type=str, help="Name of CXR data folder")
    parser.add_argument("--model_name", type=str, choices=["truncated_b0", "truncated_b0_act1", "truncated_b0_leaky", "truncated_b0_leaky2"], help="Custom model found in custom_lib.custom_models.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--rotate_angle", type=float, default=None, help="Rotation angle for data augmentation.")
    parser.add_argument("--horizontal_flip_prob", type=float, default=None, help="Probability of horizontal flip for data augmentation.")
    parser.add_argument("--brightness_contrast_range", 
    type=float, nargs=4, default=None, metavar=("BRIGHTNESS_MIN", "BRIGHTNESS_MAX", "CONTRAST_MIN", "CONTRAST_MAX"),
    help="Brightness and contrast range as four float values: brightness_min brightness_max contrast_min contrast_max"
    )
    parser.add_argument("--gaussian_blur", 
    type=int, nargs=2, default=None, metavar=("kernel_size", "sigma"),
    help="Specifies the parameters for Gaussian blur. 'kernel_size' defines the size of the filter (i.e., how many pixels are considered during the blur), and 'sigma' controls the amount of blur applied (i.e., the standard deviation of the Gaussian distribution)."
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize the data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--truncated_layers", type=int, default=0, help="Number of layers to truncate from EfficientNet.")
    parser.add_argument("--results_folder_name", type=str, help="Folder to save results.")
    parser.add_argument("--save_logs", action="store_true", help="Save logs and outputs.")
    parser.add_argument("--image_size", type=int, default=224, help="Size of image for resize in data transform")
    parser.add_argument("--center_crop", type = int, default=224, help="Centercrop of image in data transform")
    parser.add_argument("--dropout_p", type=float, default=.2, help="The probablity for the dropout in classifier layer.")




    args = parser.parse_args()
    main(args)