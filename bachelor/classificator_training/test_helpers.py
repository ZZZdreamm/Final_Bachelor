from collections import defaultdict
import torch
import torch.nn.functional as F
from classificator_training.utils import move_to_device


BOLD_GREEN = '\033[1;32m'
BOLD_RED = '\033[1;31m'
RESET = '\033[0m'


def generate_class_prototypes(model, dataloader, device, max_batches=None):
    """
    Generates the mean embedding for each class from the training data.
    """
    model.eval()

    class_embeddings = {}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            if max_batches is not None and i >= max_batches:
                break

            inputs = batch['anchor']
            labels = batch['y']
            
            embeddings = model(**inputs)
            
            for embed, label in zip(embeddings, labels):
                if label.item() not in class_embeddings:
                    class_embeddings[label.item()] = [embed]
                else:
                    class_embeddings[label.item()].append(embed)

    prototypes = {}
    
    for class_id, embed_list in class_embeddings.items():
        stacked_embeddings = torch.stack(embed_list)
        mean_prototype = torch.mean(stacked_embeddings, dim=0)
        
        normalized_prototype = F.normalize(mean_prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
        prototypes[class_id] = normalized_prototype

    class_ids = sorted(prototypes.keys())
    prototype_tensor = torch.stack([prototypes[c] for c in class_ids]).to(device)
    
    return prototype_tensor, class_ids

def evaluate_ncm_accuracy_top3(model, dataloader, prototypes, class_ids, id_to_tag, device):
    """
    Evaluates the model on the test set using the Nearest Class Mean (NCM) classifier
    and calculates both Top-1 and Top-3 accuracy.
    """
    model.eval()
    class_top1_correct_predictions = defaultdict(int)
    class_top3_correct_predictions = defaultdict(int) 
    class_total_samples = defaultdict(int)

    total_top1_correct_predictions = 0
    total_top3_correct_predictions = 0 
    total_samples = 0

    prototypes = prototypes.to(device)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = move_to_device(batch, device) 

            T_inputs = batch['anchor']
            true_labels = batch['y']

            test_embeddings = model(**T_inputs)
            similarity_matrix = test_embeddings @ prototypes.T

            predicted_class_indices_top1 = torch.argmax(similarity_matrix, dim=1)
            predicted_labels_top1 = torch.tensor([class_ids[idx.item()] 
                                                 for idx in predicted_class_indices_top1], device=device)
            
            is_correct_top1 = (predicted_labels_top1 == true_labels)

            K = 3
            topK_results = torch.topk(similarity_matrix, K, dim=1)
            predicted_class_indices_topK = topK_results.indices
            topk_predicted_labels = torch.tensor([[class_ids[idx.item()] for idx in row] 
                                                 for row in predicted_class_indices_topK], device=device)
            
            is_in_topk = (topk_predicted_labels == true_labels.unsqueeze(1).expand(-1, K))
            is_correct_top3 = torch.any(is_in_topk, dim=1)

            total_samples += true_labels.size(0)
            total_top1_correct_predictions += is_correct_top1.sum().item()
            total_top3_correct_predictions += is_correct_top3.sum().item()

            for true_label, is_corr1, is_corr3 in zip(true_labels.tolist(), 
                                                     is_correct_top1.tolist(), 
                                                     is_correct_top3.tolist()):
                class_total_samples[true_label] += 1
                if is_corr1:
                    class_top1_correct_predictions[true_label] += 1
                if is_corr3:
                    class_top3_correct_predictions[true_label] += 1

    total_top1_accuracy = total_top1_correct_predictions / total_samples if total_samples > 0 else 0.0
    total_top3_accuracy = total_top3_correct_predictions / total_samples if total_samples > 0 else 0.0

    print(f"\nTest Results:")
    print(f"  Overall Top-1 Accuracy: {total_top1_accuracy:.4f} ({total_top1_correct_predictions}/{total_samples})")
    print(f"  Overall Top-3 Accuracy: {total_top3_accuracy:.4f} ({total_top3_correct_predictions}/{total_samples})")

    all_class_ids = sorted(class_total_samples.keys())

    top1_classes_above_50 = []
    top1_count_above_50 = 0

    for class_id in all_class_ids:
        corr_top1 = class_top1_correct_predictions[class_id] 
        total = class_total_samples[class_id]
        accuracy_top1 = corr_top1 / total if total > 0 else 0.0
        class_tag = id_to_tag.get(class_id, f"Unknown ID {class_id}")

        if accuracy_top1 >= 0.50:
            top1_classes_above_50.append((class_tag, accuracy_top1, corr_top1, total))
            top1_count_above_50 += 1
    classes_above_50 = []
    classes_below_50 = []
    count_above_50 = 0

    for class_id in all_class_ids:
        corr_top3 = class_top3_correct_predictions[class_id]
        total = class_total_samples[class_id]
        accuracy_top3 = corr_top3 / total if total > 0 else 0.0
        class_tag = id_to_tag.get(class_id, f"Unknown ID {class_id}")

        if accuracy_top3 >= 0.50:
            classes_above_50.append((class_tag, accuracy_top3, corr_top3, total))
            count_above_50 += 1
        else:
            classes_below_50.append((class_tag, accuracy_top3, corr_top3, total))

    print(f"\nClasses with Top-3 Accuracy >= 50%: {count_above_50}/{len(all_class_ids)}")
    for class_tag, acc, corr, total in classes_above_50[:5]:
        print(f"  {class_tag}: {acc:.2%} ({corr}/{total})")

    return total_top1_accuracy, total_top3_accuracy