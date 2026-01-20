import cv2
import numpy as np
import json
import os
import glob

IMAGE_FOLDER = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_przetworzone"
OUTPUT_JSON = "test_dataset_annotations.json"

class Annotator:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise ValueError(f"Nie można wczytać zdjęcia: {img_path}")
            
        self.h, self.w = self.img.shape[:2]
        self.window_name = "Annotator Dynamic (Wymiary + Pozycja)"
        
        cv2.namedWindow(self.window_name)
        
        cv2.createTrackbar('Pan X', self.window_name, 500, 1000, self.nothing)
        cv2.createTrackbar('Pan Y', self.window_name, 500, 1000, self.nothing)
        cv2.createTrackbar('Dist Z', self.window_name, 500, 2000, self.nothing) 
        cv2.createTrackbar('Focal', self.window_name, 1000, 3000, self.nothing) 
        
        cv2.createTrackbar('Pitch (X)', self.window_name, 180 + 90, 360, self.nothing) 
        cv2.createTrackbar('Yaw (Y)', self.window_name, 180, 360, self.nothing)   
        cv2.createTrackbar('Roll (Z)', self.window_name, 180, 360, self.nothing)  
        
        cv2.createTrackbar('Size W (X)', self.window_name, 100, 500, self.nothing) 
        cv2.createTrackbar('Size H (Y)', self.window_name, 100, 500, self.nothing) 
        cv2.createTrackbar('Size D (Z)', self.window_name, 100, 500, self.nothing) 

        cv2.setTrackbarPos('Pan X', self.window_name, 500)
        cv2.setTrackbarPos('Pan Y', self.window_name, 500)

    def nothing(self, x):
        pass

    def get_model_vertices(self):
        """Generuje wierzchołki 3D na podstawie aktualnych suwaków wymiarów."""
        w = cv2.getTrackbarPos('Size W (X)', self.window_name)
        h = cv2.getTrackbarPos('Size H (Y)', self.window_name)
        d = cv2.getTrackbarPos('Size D (Z)', self.window_name)
        
        x = w / 2.0
        y = h / 2.0
        z = d / 2.0
        
        verts = np.array([
            [-x, -y, -z], # 0: Lewy-Góra-Tył
            [-x, -y,  z], # 1: Lewy-Góra-Przód
            [-x,  y, -z], # 2: Lewy-Dół-Tył
            [-x,  y,  z], # 3: Lewy-Dół-Przód
            [ x, -y, -z], # 4: Prawy-Góra-Tył
            [ x, -y,  z], # 5: Prawy-Góra-Przód
            [ x,  y, -z], # 6: Prawy-Dół-Tył
            [ x,  y,  z], # 7: Prawy-Dół-Przód
        ], dtype=np.float32)
        
        return verts

    def get_projected_points(self, vertices_3d):
        px = (cv2.getTrackbarPos('Pan X', self.window_name) - 500) * 2 + self.w/2
        py = (cv2.getTrackbarPos('Pan Y', self.window_name) - 500) * 2 + self.h/2
        dist_z = cv2.getTrackbarPos('Dist Z', self.window_name) + 10
        f = cv2.getTrackbarPos('Focal', self.window_name)

        rx = np.radians(cv2.getTrackbarPos('Pitch (X)', self.window_name) - 180)
        ry = np.radians(cv2.getTrackbarPos('Yaw (Y)', self.window_name) - 180)
        rz = np.radians(cv2.getTrackbarPos('Roll (Z)', self.window_name) - 180)

        K = np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float32)

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        t = np.array([[0], [0], [dist_z]], dtype=np.float32)

        rvec, _ = cv2.Rodrigues(R)
        img_points, _ = cv2.projectPoints(vertices_3d, rvec, t, K, distCoeffs=None)
        return img_points.reshape(-1, 2)

    def run(self):
        edges = [
            (0,1), (1,3), (3,2), (2,0), 
            (4,5), (5,7), (7,6), (6,4), 
            (0,4), (1,5), (2,6), (3,7)  
        ]

        while True:
            display_img = self.img.copy()
            
            verts_3d = self.get_model_vertices()
            
            points_2d = self.get_projected_points(verts_3d)

            for s, e in edges:
                pt1 = tuple(points_2d[s].astype(int))
                pt2 = tuple(points_2d[e].astype(int))
                cv2.line(display_img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

            for i, pt in enumerate(points_2d):
                color = (0, 0, 255) 
                if verts_3d[i][2] > 0: 
                    color = (255, 0, 0) 
                
                cv2.circle(display_img, tuple(pt.astype(int)), 5, color, -1)

            cv2.putText(display_img, "Dopasuj KSZTALT i POZYCJE (S: Zapisz, ESC: Wyjdz)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: 
                return None
            elif key == ord('s'):
                print(f"Zapisano układ dla: {os.path.basename(self.window_name)}")
                return points_2d.tolist()

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Utwórz folder '{IMAGE_FOLDER}' i wrzuć tam zdjęcia.")
        return

    exts = ['*.png', '*.jpg', '*.jpeg']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
    images = sorted(images)
    
    dataset = []
    
    print(f"Znaleziono {len(images)} zdjęć.")
    
    for img_path in images:
        print(f"Edycja: {img_path}")
        try:
            annotator = Annotator(img_path)
            points = annotator.run()
        except Exception as e:
            print(f"Błąd przy {img_path}: {e}")
            continue
        
        if points is None:
            print("Przerwano przez użytkownika.")
            break
            
        dataset.append({
            "filename": os.path.basename(img_path),
            "corners_2d": points
        })
        
        cv2.destroyAllWindows()

    if dataset:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Sukces! Zapisano {len(dataset)} adnotacji w pliku {OUTPUT_JSON}")

if __name__ == "__main__":
    main()