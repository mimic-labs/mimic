import numpy as np
from sklearn.decomposition import PCA
import cv2

def generate_random_embeddings(num_embeddings, embedding_dim):
    embeddings = np.random.randn(num_embeddings, embedding_dim)
    return embeddings

def generate_random_positions(num):
    min_width, max_width = 40, 150

    x_tl = np.random.randint(0, 1280 - max_width, size=num)
    y_tl = np.random.randint(0, 720 - max_width, size=num)
    z = np.random.randint(0, 400, size=num)

    widths = np.random.randint(min_width, max_width, size=num) 
    heights = np.random.randint(min_width, max_width, size=num)

    x_br = x_tl + widths  
    y_br = y_tl + heights  
    x_tr = x_tl + widths   
    y_tr = y_tl           
    x_bl = x_tl          
    y_bl = y_tl + heights 

    bounding_boxes = np.column_stack((x_tl, y_tl, z, x_br, y_br, z, x_tr, y_tr, z, x_bl, y_bl, z))
    return bounding_boxes

def generate_sequence(embeddings, positions, num, num_captures):
    camera_shifts_x = np.random.randint(0, 200, size=num_captures)    
    camera_shifts_y = np.random.randint(0, 200, size=num_captures)
    
    positions = np.repeat(positions[None, :, :], num_captures, axis=0)
    positions[:, :, 0] += camera_shifts_x[:, None]  
    positions[:, :, 3] += camera_shifts_x[:, None]  
    positions[:, :, 6] += camera_shifts_x[:, None]
    positions[:, :, 9] += camera_shifts_x[:, None] 

    positions[:, :, 1] += camera_shifts_y[:, None]
    positions[:, :, 4] += camera_shifts_y[:, None]
    positions[:, :, 7] += camera_shifts_y[:, None]
    positions[:, :, 10] += camera_shifts_y[:, None]

    # TODO: add dropout of some positions with a probability
    # skip for now

    ordering = np.random.randint(0, num, size=num_captures)
    # true_embeddings_ordered = embeddings[ordering, :]

    hand_pos = np.zeros((num_captures, 12))
    for i in range(ordering.shape[0]):
        hand_pos[i, :] = positions[i, ordering[i], :]
    
    # TODO: generate embeddings for each position with some noise
    embeddings_recorded = np.repeat(embeddings[None, :, :], num_captures, axis=0)
    embeddings_noise = np.random.randn(*embeddings_recorded.shape) * 0.03
    embeddings_recorded = embeddings_recorded + embeddings_noise

    # TODO: consolidate hand_pos to hand center and add some noise
    hand_pos = hand_pos.reshape(hand_pos.shape[0], 4, 3)
    hand_pos = np.mean(hand_pos, axis=1)
    
    hand_xy_shift = np.random.uniform(0, 100, size=(num_captures,2))
    hand_z_shift = np.random.uniform(0, 50, size=(num_captures,1))
    hand_shift = np.concatenate((hand_xy_shift, hand_z_shift), axis=1)
    hand_pos = hand_pos + hand_shift

    return hand_pos, positions, embeddings_recorded, ordering


def generate_paired_captures(ordering, embeddings, num):
    base_image_positions = generate_random_positions(num)
    img_pos = base_image_positions[np.newaxis, :, :]

    embeddings_noise = np.random.randn(*embeddings.shape) * 0.03
    base_image_embeddings = embeddings + embeddings_noise
    img_embeddings = base_image_embeddings[np.newaxis, :, :]

    correct_paired_hand_pos = base_image_positions[ordering, :]
    correct_paired_hand_pos = correct_paired_hand_pos.reshape(correct_paired_hand_pos.shape[0], 4, 3)
    correct_paired_hand_pos = np.mean(correct_paired_hand_pos, axis=1)

    return img_embeddings, img_pos, correct_paired_hand_pos

def visualize_captures(embeddings, positions, hand_pos, numCaptures, numObjects, train=True):
    for i in range(numCaptures):
        img = np.zeros((1200, 1700, 3), dtype=int) 
        for j in range(numObjects):
            embedding = embeddings[i, j, :]
            pca = PCA(n_components=3)
            reduced_embedding = pca.fit_transform(embedding.reshape(4, 32))
            normalized_embedding = (reduced_embedding - reduced_embedding.min()) / (reduced_embedding.max() - reduced_embedding.min())
            r, g, b = normalized_embedding[0]
            color = np.array([int(r * 255), int(g * 255), int(b * 255)])

            corners = positions[i, j, :].reshape((4, 3))
            width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            height = np.max(corners[:, 1]) - np.min(corners[:, 1])
            y, x = corners[0, 1], corners[0, 0]
            img[y:y+height, x:x+width, :] = color
        
        center = (int(hand_pos[i, 0]), int(hand_pos[i, 1]))
        img = img.astype(np.uint8)
        cv2.circle(img, center, 10, (0, 255, 255), -1)

        if train:
            label = f"Train Capture {i}"
        else:
            label = f"Test Base Capture"

        cv2.imshow(label, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    num_objects = 10
    num_captures = 5
    embeddings = generate_random_embeddings(num_objects, 128)
    positions = generate_random_positions(num_objects)
    hand_pos, positions, embeddings_recorded, ordering = generate_sequence(embeddings, positions, num_objects, num_captures)
    base_img_embeddings, base_img_positions, correct_hand_pos = generate_paired_captures(ordering, embeddings, num_objects)

    visualize_captures(embeddings_recorded, positions, hand_pos, num_captures, num_objects, train=True)
    visualize_captures(base_img_embeddings, base_img_positions, correct_hand_pos, num_captures, num_objects, train=False)

    # embeddings = (num_captures, num_objects, 128)
    # positions = (num_captures, num_objects, 12)
    # hand_pos = (num_captures, 3)