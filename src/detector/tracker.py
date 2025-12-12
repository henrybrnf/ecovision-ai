import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class TrackedObject:
    id: int
    centroid: Tuple[int, int]
    bbox: List[int]
    confidence: float
    velocity: float = 0.0
    velocity_vector: Tuple[float, float] = (0.0, 0.0) # Vector (dx, dy) para predicción
    last_frame: int = 0 # Frame donde fue detectado
    history: List[Tuple[int, int]] = None
    frames_missed: int = 0
    is_running: bool = False
    class_id: int = 0
    
    @property
    def center(self):
        return self.centroid

class SimpleTracker:
    def __init__(self, max_dist=150, max_missed=10): # max_dist aumentado para seguir a gente corriendo
        self.next_id = 0
        self.objects: Dict[int, TrackedObject] = {}
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.speed_threshold = 12.0 # Aumentado a 12.0 para evitar que caminata rápida sea "Running"

    def update(self, detections, frame_count=0) -> List[TrackedObject]:
        # detections: lista de objetos Detection(bbox, confidence, class_id, center)
        
        input_centroids = [d.center for d in detections]
        input_bboxes = [d.bbox for d in detections]
        input_confs = [getattr(d, 'confidence', 0.0) for d in detections]
        input_classes = [getattr(d, 'class_name', "person") for d in detections]
        
        if len(input_centroids) == 0:
            # Nadie detectado, aumentar contador de perdidos
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].frames_missed += 1
                if self.objects[obj_id].frames_missed > self.max_missed:
                    del self.objects[obj_id]
            return []

        # Si no hay objetos rastreados, registrar todos
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], input_confs[i], input_classes[i], frame_count)
            return list(self.objects.values())

        # Match de centroides
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]

        # Matriz de distancias
        D = np.zeros((len(object_ids), len(input_centroids)))
        for i in range(len(object_ids)):
            for j in range(len(input_centroids)):
                dist = np.linalg.norm(np.array(object_centroids[i]) - np.array(input_centroids[j]))
                D[i, j] = dist

        # Encontrar pares mínimos
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue

            if D[r, c] > self.max_dist:
                continue

            obj_id = object_ids[r]
            self._update_object(obj_id, input_centroids[c], input_bboxes[c], input_confs[c], frame_count)

            used_rows.add(r)
            used_cols.add(c)

        # Registrar nuevos
        for c in range(len(input_centroids)):
            if c not in used_cols:
                self._register(input_centroids[c], input_bboxes[c], input_confs[c], input_classes[c], frame_count)


        # Eliminar perdidos
        for r in range(len(object_ids)):
            if r not in used_rows:
                obj_id = object_ids[r]
                self.objects[obj_id].frames_missed += 1
                if self.objects[obj_id].frames_missed > self.max_missed:
                    del self.objects[obj_id]

        return list(self.objects.values())

    def _register(self, centroid, bbox, confidence, class_name, frame_count):
        self.objects[self.next_id] = TrackedObject(
            id=self.next_id,
            centroid=centroid,
            bbox=bbox,
            confidence=confidence,
            last_frame=frame_count,
            history=[centroid],
            class_id=1 if class_name == "pattern" else 0
        )
        self.next_id += 1

    def _update_object(self, obj_id, new_centroid, new_bbox, confidence, frame_count):
        obj = self.objects[obj_id]
        
        # Calcular vector de velocidad (dx, dy)
        prev_centroid = obj.centroid
        dx = new_centroid[0] - prev_centroid[0]
        dy = new_centroid[1] - prev_centroid[1]
        
        # Calcular magnitud
        dist = np.sqrt(dx*dx + dy*dy)
        
        # EMA para magnitud (Suavizado más agresivo para evitar picos de ruido)
        # alpha = 0.3 (antes 0.7) da más peso a la historia
        obj.velocity = 0.3 * dist + 0.7 * obj.velocity
        
        # EMA para vector
        old_vx, old_vy = obj.velocity_vector
        new_vx = 0.3 * dx + 0.7 * old_vx
        new_vy = 0.3 * dy + 0.7 * old_vy
        obj.velocity_vector = (new_vx, new_vy)
        
        # Actualizar estado
        obj.centroid = new_centroid
        obj.bbox = new_bbox
        obj.confidence = confidence
        obj.last_frame = frame_count
        obj.frames_missed = 0
        obj.history.append(new_centroid)
        if len(obj.history) > 20:
            obj.history.pop(0)
            
        # Determinar si corre
        obj.is_running = obj.velocity > self.speed_threshold

    def get_predicted_objects(self, target_frame) -> List[TrackedObject]:
        predicted = []
        for obj in self.objects.values():
            delta = target_frame - obj.last_frame
            vx, vy = obj.velocity_vector
            
            # Predecir centro
            new_cx = int(obj.centroid[0] + vx * delta)
            new_cy = int(obj.centroid[1] + vy * delta)
            
            # Desplazar bbox
            shift_x = int(vx * delta)
            shift_y = int(vy * delta)
            new_bbox = [
                obj.bbox[0] + shift_x,
                obj.bbox[1] + shift_y,
                obj.bbox[2] + shift_x,
                obj.bbox[3] + shift_y
            ]
            
            # Clonar objeto con nueva posición
            p_obj = TrackedObject(
                id=obj.id,
                centroid=(new_cx, new_cy),
                bbox=new_bbox,
                confidence=obj.confidence,
                velocity=obj.velocity,
                velocity_vector=obj.velocity_vector,
                last_frame=target_frame,
                is_running=obj.is_running,
                class_id=obj.class_id
            )
            predicted.append(p_obj)
        return predicted
