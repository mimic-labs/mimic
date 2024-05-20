import numpy as np
import torch
from sklearn.decomposition import PCA
import cv2
from model.dataset import CustomDataset

def generate_random_obj_embeddings(num_objects = 10, embedding_dim = 128):
    embeddings = torch.randn(num_objects, embedding_dim)
    return embeddings

def generate_random_obj_positions(num_objects = 10):
    min_width, max_width = 40, 150

    x_tl = torch.randint(0, 1280 - max_width, size=(num_objects,))
    y_tl = torch.randint(0, 720 - max_width, size=(num_objects,))
    z = torch.randint(0, 400, size=(num_objects,))

    widths = torch.randint(min_width, max_width, size=(num_objects,)) 
    heights = torch.randint(min_width, max_width, size=(num_objects,))

    x_br = x_tl + widths  
    y_br = y_tl + heights  
    x_tr = x_tl + widths   
    y_tr = y_tl           
    x_bl = x_tl          
    y_bl = y_tl + heights 

    bounding_boxes = torch.column_stack((x_tl, y_tl, z, x_br, y_br, z, x_tr, y_tr, z, x_bl, y_bl, z))
    return bounding_boxes

def generate_rand_obj_sequence(obj_pos_start: torch.Tensor, num_captures: int):
    camera_shifts_x = torch.randint(0, 200, size=(num_captures,))
    camera_shifts_y = torch.randint(0, 200, size=(num_captures,))
    
    obj_pos_seq = obj_pos_start[None, :, :].repeat_interleave(num_captures, dim=0)
    obj_pos_seq[:, :, 0] += camera_shifts_x[:, None]  
    obj_pos_seq[:, :, 3] += camera_shifts_x[:, None]  
    obj_pos_seq[:, :, 6] += camera_shifts_x[:, None]
    obj_pos_seq[:, :, 9] += camera_shifts_x[:, None] 

    obj_pos_seq[:, :, 1] += camera_shifts_y[:, None]
    obj_pos_seq[:, :, 4] += camera_shifts_y[:, None]
    obj_pos_seq[:, :, 7] += camera_shifts_y[:, None]
    obj_pos_seq[:, :, 10] += camera_shifts_y[:, None]
    
    return obj_pos_seq

# generate simple hand sequence (move in a straight line)
def generate_static_hand_seq(num_captures: int):
    hand_pos_seq = torch.zeros(num_captures, 1, 3) # 1 = single hand for now
    hand_pos_seq[:, 0, 0] = torch.linspace(0, 1000, num_captures)
    # hand_pos_seq[:, 1] = torch.linspace(0, 720, num_captures)
    # hand_pos_seq[:, 2] = torch.linspace(0, 400, num_captures)
    return hand_pos_seq

# generate sequence of hand positions (aka order of objects it interacts w/)
def generate_hand_seq(obj_pos_seq: torch.Tensor, ordering: torch.Tensor, num_captures: int):
    # based on ordering, set initial position of hand at each object in sequence
    hand_pos_seq = obj_pos_seq[torch.arange(num_captures), ordering, :] # select obj positions for each capture
    print(hand_pos_seq.shape)

    # consolidate hand_pos to hand center and add some noise
    hand_pos_seq = hand_pos_seq.reshape(hand_pos_seq.shape[0], 4, 3)
    hand_pos_seq = torch.mean(hand_pos_seq, axis=1, dtype=torch.float)
    print(f"{hand_pos_seq.shape=}")
    
    hand_xy_shift = torch.FloatTensor(num_captures, 2).uniform_(0, 100)
    hand_z_shift = torch.FloatTensor(num_captures, 1).uniform_(0, 50)
    hand_shift = torch.cat((hand_xy_shift, hand_z_shift), axis=1)
    
    hand_pos_seq = hand_pos_seq + hand_shift
    hand_pos_seq = hand_pos_seq.unsqueeze(1) # make it (num_captures, 1, 3) for 1 hand
    
    return hand_pos_seq

# generate combined sequence of object positions and hand positions
def generate_sequence(
    embeddings: torch.Tensor,
    obj_pos_start: torch.Tensor,
    num_objects: int = 10,
    num_captures: int = 5
):
    # generate random sequence of object positions
    obj_pos_seq = generate_rand_obj_sequence(obj_pos_start, num_captures)
    
    # TODO: add dropout of some positions with a probability
    # skip for now

    # true order of images, ground truth
    ordering = torch.randint(0, num_objects, size=(num_captures,))
    # true_embeddings_ordered = embeddings[ordering, :]

    # generate embeddings for each position with some noise
    embeddings_recorded = (embeddings[None, :, :]).repeat_interleave(num_captures, dim=0)
    embeddings_noise = torch.randn(*embeddings_recorded.shape) * 0.03
    embeddings_recorded = embeddings_recorded + embeddings_noise

    # based on ordering, set initial position of hand at each object in sequence
    hand_pos_seq = generate_static_hand_seq(num_captures)

    return hand_pos_seq, obj_pos_seq, embeddings_recorded, ordering

# generate corresponding sequence/positiond data for "target" environment
def generate_paired_captures(ordering, embeddings, num_objects, num_captures):
    base_image_positions = generate_random_obj_positions(num_objects)
    img_pos = base_image_positions[None, :, :]

    embeddings_noise = torch.randn(*embeddings.shape) * 0.03
    base_image_embeddings = embeddings + embeddings_noise
    img_embeddings = base_image_embeddings[None, :, :]
    # print(base_image_positions.shape)

    # correct_paired_hand_pos = base_image_positions[ordering, :] # orders the test images to create full ground truth
    # correct_paired_hand_pos = correct_paired_hand_pos.reshape(correct_paired_hand_pos.shape[0], 4, 3)
    # correct_paired_hand_pos = correct_paired_hand_pos.mean(axis=1, dtype=torch.float)


    # img_pos = img_pos.repeat_interleave(len(ordering), dim=0)
    img_embeddings = img_embeddings.repeat_interleave(len(ordering), dim=0)
    img_pos = generate_rand_obj_sequence(base_image_positions, num_captures)
    
    # correct_paired_hand_pos = generate_hand_seq(img_pos, ordering, num_captures)
    correct_paired_hand_pos = generate_static_hand_seq(num_captures)
    
    # print(correct_paired_hand_pos.shape)
    # print(img_pos.shape)
    # print(img_embeddings.shape)
    
    
    # img pos is new/target object sequence
    return correct_paired_hand_pos, img_pos, img_embeddings

    # return correct_paired_hand_pos, img_pos, img_embeddings

# display positions of objects and hand in each capture
def visualize_captures(
    num_captures,
    num_objects,
    embeddings,
    positions,
    true_hand_pos,
    pred_hand_pos=None,
    train=True
):
    for i in range(num_captures):
        img = torch.zeros((1200, 1700, 3), dtype=int) 
        for j in range(num_objects):
            embedding = embeddings[i, j, :]
            pca = PCA(n_components=3)
            reduced_embedding = pca.fit_transform(embedding.reshape(4, 32))
            normalized_embedding = (reduced_embedding - reduced_embedding.min()) / (reduced_embedding.max() - reduced_embedding.min())
            r, g, b = normalized_embedding[0]
            color = torch.Tensor([int(r * 255), int(g * 255), int(b * 255)])

            corners = positions[i, j, :].reshape(4, 3)
            width = torch.max(corners[:, 0]) - torch.min(corners[:, 0])
            height = torch.max(corners[:, 1]) - torch.min(corners[:, 1])
            y, x = corners[0, 1], corners[0, 0]
            img[y:y+height, x:x+width, :] = color
        
        img = img.numpy().astype(np.uint8)
        
        true_hand_center = (int(true_hand_pos[i, 0, 0]), int(true_hand_pos[i, 0, 1]))
        cv2.circle(img, true_hand_center, 10, (0, 255, 0), -1)
        
        if pred_hand_pos is not None:
            pred_hand_center = (int(pred_hand_pos[i, 0, 0]), int(pred_hand_pos[i, 0, 1]))
            cv2.circle(img, pred_hand_center, 10, (255, 165, 0), -1)

        if train:
            label = f"Train Capture {i}"
        else:
            label = f"Test Base Capture"
            
        cv2.putText(
            img=img,
            text=label,
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2
        )

        # cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)          
        # cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output",img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# generate training data for transformer model
def generate_train_data(num_captures, num_objects, num_examples):
    all_embeddings_recorded = []
    all_ref_object_positions = []
    all_ref_hand_pos = []
    all_new_object_embeddings = []
    all_new_object_positions = []
    all_correct_hand_pos = []
    
    for i in range(num_examples):
        obj_embeddings = generate_random_obj_embeddings(num_objects, 128)
        obj_positions = generate_random_obj_positions(num_objects)
        
        hand_pos, obj_positions, embeddings_recorded, ordering = generate_sequence(obj_embeddings, obj_positions, num_objects, num_captures)
        correct_hand_pos, base_img_positions, base_img_embeddings = generate_paired_captures(ordering, obj_embeddings, num_objects, num_captures)
        
        all_embeddings_recorded.append(embeddings_recorded)
        all_ref_object_positions.append(obj_positions)
        all_ref_hand_pos.append(hand_pos)
        all_new_object_embeddings.append(base_img_embeddings)
        all_new_object_positions.append(base_img_positions)
        all_correct_hand_pos.append(correct_hand_pos)

    return CustomDataset(
        embeddings=torch.stack(all_embeddings_recorded),
        ref_object_positions=torch.stack(all_ref_object_positions),
        ref_hand_pos=torch.stack(all_ref_hand_pos),
        new_object_embeddings=torch.stack(all_new_object_embeddings),
        new_object_positions=torch.stack(all_new_object_positions),
        correct_hand_pos=torch.stack(all_correct_hand_pos)
    )

    # return CustomDataset(
    #     embeddings=embeddings_recorded,
    #     ref_object_positions=obj_positions,#.astype(np.float64),
    #     ref_hand_pos=hand_pos,
    #     new_object_embeddings=base_img_embeddings,
    #     new_object_positions=base_img_positions,#.astype(np.float64),
    #     correct_hand_pos=correct_hand_pos
    # )


if __name__ == "__main__":
    num_objects = 10
    num_captures = 5
    
    obj_embeddings = generate_random_obj_embeddings(num_objects, 128)
    obj_positions = generate_random_obj_positions(num_objects)
    
    hand_pos, obj_positions, embeddings_recorded, ordering = generate_sequence(obj_embeddings, obj_positions, num_objects, num_captures)
    correct_hand_pos, base_img_positions, base_img_embeddings = generate_paired_captures(ordering, obj_embeddings, num_objects, num_captures)

    visualize_captures(num_captures, num_objects, embeddings_recorded, obj_positions, hand_pos, train=True)
    visualize_captures(num_captures, num_objects, base_img_embeddings, base_img_positions, correct_hand_pos, train=False) # test only has one start capture

    # embeddings = (num_captures, num_objects, 128)
    # positions = (num_captures, num_objects, 12)
    # hand_pos = (num_captures, 3)