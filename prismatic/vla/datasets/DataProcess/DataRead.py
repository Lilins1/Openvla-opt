import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os
import numpy as np
from lineFit import fitBezierToolBox 
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import dct,idct
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Add at top

class DataRead:
    # 设置路径
    base_path = r"E:\Software\VLA\Data\modified_libero_rlds\libero_10_no_noops\1.0.0"
    action_sequences = []

    def data_struct(self):
        # 找到所有 TFRecord 文件
        tfrecord_files = glob.glob(os.path.join(self.base_path, "*.tfrecord-*"))
        print(f"Found {len(tfrecord_files)} TFRecord files.")

        # 创建 TFRecordDataset
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

        # 取第一条数据示例
        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            feature_dict = example.features.feature
            self.print_struct(feature_dict)
            feature = feature_dict['steps/action']
            kind = feature.WhichOneof("kind")  # 判断是哪种类型
            actions = getattr(feature, kind).value
            self.action_sequences = self.action_to_list(actions,dim = 7)
            dimension = []
            for index in range(7):
                dimension.append(fitBezierToolBox.dots_into_bezier(dots = [seq[index] for seq in self.action_sequences]))

            # print(dimension[6])



            

        # # 统计所有 Example 条数
        # count = sum(1 for _ in raw_dataset)
        # print("Number of records:", count)

    def data_print(self):...

    def print_struct(self,feature_dict):
                # 需要读取的字段列表
        keys = [
            'steps/is_first',
            'steps/action',
            'steps/discount',
            'steps/is_last',
            'steps/language_instruction',
            'steps/observation/wrist_image',
            'steps/reward',
            'steps/is_terminal',
            'steps/observation/state',
            'steps/observation/joint_state',
            'steps/observation/image',
            'episode_metadata/file_path'
        ]

        for key in keys:
            if key in feature_dict:
                feature = feature_dict[key]
                kind = feature.WhichOneof("kind")  # 判断是哪种类型
                value = getattr(feature, kind).value
                # 打印key和对应值的简单信息
                key = key + " "*(30-len(key))
                if kind == 'bytes_list':
                    print(f"{key}: bytes_list with {len(value)} elements")
                elif kind == 'float_list':
                    print(f"{key}: float_list with {len(value)} elements, values example: {[f'{v:.3f}' for v in value[:14]]}")
                elif kind == 'int64_list':
                    print(f"{key}: int64_list with {len(value)} elements, values example: {[f'{v:.3f}' for v in value[:7]]}")
                else:
                    print(f"{key}: unknown feature type")
            else:
                print(f"{key} not found in features")

    def action_to_list(self,actions,dim = 7):
        # actions = np.array(actions, dtype=float)
        length = len(actions)
        index = 0
        sequences = []
        while index + dim <= length:
            action = actions[index:index+7]
            sequences.append(action)
            index += dim
        return sequences

    def action_to_bezier(self,sequences,epsilon):
        self.line = fitBezierToolBox.fit_beziers(sequences,epsilon)

    def plot_bezier_curves(self, bezier_curves, actions):
        """
        Plot fitted Bézier curves and original actions data.
        
        Parameters:
        - bezier_curves: List of [P0, P1, P2, length], where P0, P1, P2 are np.ndarray scalars, length is an int.
        - actions: List of original action values (scalars or 1D arrays).
        """
        # Prepare x and y for original actions
        action_x = np.arange(len(actions))
        action_y = [action[0] if isinstance(action, (list, np.ndarray)) else action for action in actions]

        # Initialize lists for Bézier curve points
        bezier_x = []
        bezier_y = []
        current_x = 0  # Starting x position

        # Process each Bézier curve segment
        for curve in bezier_curves:
            P0, P1, P2, length = curve
            # Extract scalar values from NumPy arrays
            P0 = P0.item() if isinstance(P0, np.ndarray) else P0
            P1 = P1.item() if isinstance(P1, np.ndarray) else P1
            P2 = P2.item() if isinstance(P2, np.ndarray) else P2
            # Generate t values from 0 to 1
            t = np.linspace(0, 1, 100)
            # Compute points on the quadratic Bézier curve
            y = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
            # Map t to global x-axis based on length
            x = current_x + t * length
            bezier_x.extend(x)
            bezier_y.extend(y)
            # Update the starting x for the next segment
            current_x += length

        # Plot original actions as red scatter points
        plt.scatter(action_x, action_y, color='red', label='Original Actions', zorder=5)

        # Plot Bézier curves as a blue line
        plt.plot(bezier_x, bezier_y, color='blue', label='Fitted Bézier Curve')

        # Add labels and legend
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.title('Bézier Curve Fit vs Original Actions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def gen_bezier_curves(self, bezier_curves):

        # Initialize lists for Bézier curve points
        bezier_x = []
        bezier_y = []
        current_x = 0  # Starting x position

        # Process each Bézier curve segment
        for curve in bezier_curves:
            P0, P1, P2, length = curve
            # Extract scalar values from NumPy arrays
            P0 = P0.item() if isinstance(P0, np.ndarray) else P0
            P1 = P1.item() if isinstance(P1, np.ndarray) else P1
            P2 = P2.item() if isinstance(P2, np.ndarray) else P2
            # Generate t values from 0 to 1
            t = np.linspace(0, 1, 100)
            # Compute points on the quadratic Bézier curve
            y = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
            # Map t to global x-axis based on length
            x = current_x + t * length
            bezier_x.extend(x)
            bezier_y.extend(y)
            # Update the starting x for the next segment
            current_x += length

        return bezier_x,bezier_y



    def DCT_construct(self, actions, length, fun, line):

        # Step 1: Flatten the 2D list into a 1D array
        dim_actions = np.array([action[0] for action in actions])

        # Step 2: Apply DCT transformation
        dct_actions = dct(dim_actions, type=2, norm='ortho')
        # print("DCT coefficients:", dct_actions)

        # Step 3: Remove high-frequency components (keep only the first N coefficients)
        N = int(length) # Number of low-frequency components to retain
        dct_low_freq = dct_actions.copy()
        dct_low_freq[N:] = 0

        # Step 3.5: Determine quantization resolution (1/100 of max amplitude)
        max_amp = np.max(np.abs(dct_actions))
        resolution = max_amp / 100.0
        print(f"Quantization resolution set to: {resolution:.6f}")

        # Quantize low-frequency DCT coefficients
        dct_quantized = np.round(dct_low_freq / resolution) * resolution

        # Step 4: Reconstruct the signal using inverse DCT
        reconstructed = idct(dct_low_freq, type=2, norm='ortho')
        reconstructed_quant = idct(dct_quantized, type=2, norm='ortho')

        # Step 4.5: Error calculation for DCT
        mse = mean_squared_error(dim_actions, reconstructed)
        mae = mean_absolute_error(dim_actions, reconstructed)
        mse_q = mean_squared_error(dim_actions, reconstructed_quant)
        mae_q = mean_absolute_error(dim_actions, reconstructed_quant)
        print(f"Reconstruction DCT MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"Quantized DCT Reconstruction MSE: {mse_q:.6f}, MAE: {mae_q:.6f}")

        # Step 5: Compute Bézier curve and its error
        action_x = np.arange(len(actions))
        # Ensure actions as flat values
        action_y = dim_actions
        bezier_x, bezier_y = fun(line)
        # Interpolate bezier y-values at integer sample points
        bezier_interp = np.interp(action_x, bezier_x, bezier_y)
        mse_bezier = mean_squared_error(dim_actions, bezier_interp)
        mae_bezier = mean_absolute_error(dim_actions, bezier_interp)
        print(f"Bézier Curve MSE: {mse_bezier:.6f}, MAE: {mae_bezier:.6f}")


        print(f"Rate DCT/Bézier : {(mse_q/mse_bezier):.6f}, MAE: {(mae_q/mae_bezier):.6f}")


        Flag = 3
        i = 1
        # Step 6: Plot the results
        plt.figure(figsize=(12, 10))

        if Flag > 3 :
            plt.subplot(Flag, 1, i)
            plt.plot(dim_actions, label='Original Action Sequence')
            plt.legend()
            i+=1


        plt.subplot(Flag, 1, i)
        plt.plot(dct_actions, label='DCT Frequency Domain')
        plt.axvline(x=N, color='red', linestyle='--', label=f'Keep First {N}')
        plt.legend()
        i+=1

        
        plt.subplot(Flag, 1, i)
        plt.scatter(action_x, action_y, color='red', label='Original', zorder=2)
        plt.plot(reconstructed, label='Reconstructed (Low-Freq)')
        plt.legend()
        i+=1

        if Flag > 4 :
            plt.subplot(Flag, 1, i)
            plt.scatter(action_x, action_y, color='red', label='Original', zorder=2)
            plt.plot(reconstructed_quant, label='Reconstructed (Low-Freq Quantized DCT)')
            plt.legend()
            i+=1

        plt.subplot(Flag, 1, i)
        plt.scatter(action_x, action_y, color='red', label='Original', zorder=2)
        plt.plot(bezier_x, bezier_y, color='blue', label='Bézier Curve')
        plt.legend()
        i+=1

        plt.tight_layout()
        plt.show()
        return mse,mse_bezier , mae,mae_bezier




if __name__ == "__main__":
    Read = DataRead()
    Read.data_struct()
    a = 0
    b = 0
    c = 0
    d = 0
    N = 7
    for n in range(7):
        actions = [[Read.action_sequences[t][i + n] for i in range(1)] for t in range(30)]
        # print(actions)
        # print(len(actions))
        dim_actions = [action[0] for action in actions]
        max_action = dim_actions[np.argmax(dim_actions)]
        # print("Max action: " + str(max_action))
        
        Read.action_to_bezier(actions,0.1 * max_action)
        
        
        # print(Read.line)
        # print(len(actions))

        # print(len(Read.line))

        # Read.plot_bezier_curves(Read.line,actions)

        ratio = 2

        e,f,g,h = Read.DCT_construct(actions,ratio * len(Read.line),Read.gen_bezier_curves,Read.line)
        a += e
        b += f
        c += g
        d += h

    print(f"All error Rate DCT/Bézier : {(a/b):.6f}, MAE: {(c/d):.6f}")

    'All error Rate DCT/Bézier : 9.442504, MAE: 1.927605'
    '总误差更小'






        
