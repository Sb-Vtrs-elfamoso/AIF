

def find_one_example_per_class(dataloader):
    class_examples = {}
    total_classes = len(dataloader.dataset.classes)  # Assuming the dataset knows the classes

    # Loop over the DataLoader
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            label = label.item()  # Convert label tensor to integer

            # Check if this class is already collected
            if label not in class_examples:
                class_examples[label] = image  # Store the image

            # Check if we have collected one example for each class
            if len(class_examples) == total_classes:
                break
        else:
            # Continue outer loop if inner loop wasn't broken
            continue
        # Outer loop break if inner loop was broken
        break

    return class_examples

def find_examples_per_class(dataloader, selected_classes, labs, examples_per_class):
    class_examples = {}
    for cls in selected_classes:
        class_examples[labs[cls]] = []

    # Loop over the DataLoader
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            label = label.item()  # Convert label tensor to integer

            # Check if this class is one of the selected and if more examples are needed
            if label in selected_classes and len(class_examples[labs[label]]) < examples_per_class:
                class_examples[labs[label]].append(image)  # Store the image

            # Check if all selected classes have the required number of examples
            if all(len(class_examples[labs[cls]]) >= examples_per_class for cls in selected_classes):
                break
        else:
            # Continue outer loop if inner loop wasn't broken
            continue
        # Outer loop break if inner loop was broken
        break

    return class_examples
