from torch.utils.data import DataLoader
from tqdm import tqdm


def class_accuracy(net, test_dataset, print_results=False):
    if print_results: print('Testing model accuracy...')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    net.eval()
    acc = 0
    tests = 0
    for images, labels in tqdm(test_dataloader):
        for idx, image in enumerate(images):
            image = image.unsqueeze(0)
            label = labels[idx].item()
            pred = net.predict(image)
            tests += 1
            if pred == label:
                acc += 1
    acc /= tests
    if print_results: print(f"Accuracy: {acc*100:.2f}%")
    return acc

def autoencoder(autoencoder, test_dataset, images_to_show=5):
    test_dataloader = DataLoader(test_dataset, batch_size=images_to_show, shuffle=True)
    images, _ = test_dataloader.__iter__().__next__()
    input_images = []
    output_images = []

    for image in images:
        input_images.append(image)
        image = image.unsqueeze(0)
        output = autoencoder.all(image)

        output_images.append(output[0])
    
    print(input_images, output_images)