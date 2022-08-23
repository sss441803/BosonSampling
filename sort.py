import numpy as np
import cupy as cp


class Aligner(object):

    def __init__(self, d: int, chi: int, CL: np.ndarray, CC: np.ndarray, CR: np.ndarray) -> None:
        self.d = d
        self.chi = chi

        self.CL = CL
        self.CC = CC
        self.CR = CR

        self.incL = self.get_charge_beginning_index(self.d, self.CL)
        self.sizeNewL, self.idxNewL, self.idxInvL, self.incNewL, self.cNewL = self.align_info(self.chi, self.incL)

        self.incC = self.get_charge_beginning_index(self.d, self.CC)
        
        self.incR = self.get_charge_beginning_index(self.d, self.CR)
        self.sizeNewR, self.idxNewR, self.idxInvR, self.incNewR, self.cNewR = self.align_info(self.chi, self.incR)


    # Align given data
    def align_data(self, array_type: str, data: np.ndarray):
        if type(data) is cp._core.core.ndarray:
            backend = cp
        elif type(data) is np.ndarray:
            backend = np
        else:
            raise TypeError("Data is not numpy or cupy array.")
        # idxNew makes aligned array by taking elemets from the original array.
        # Because it has to fill in empty spaces with 0, we decide to use index 0 for
        # filling 0 and the rest of the indices are increased by 1. Hence we are appending
        # a 0 to the beginning of the original array before indexing.
        if array_type in ['ll','Ll','LL']:
            return backend.append(np.zeros(1, dtype=data.dtype), data)[self.idxNewL]
        elif array_type in ['lr','Lr','LR']:
            return backend.append(np.zeros(1, dtype=data.dtype), data)[self.idxNewR]
        elif array_type in ['glc','Glc','GLC']:
            return backend.vstack( [backend.zeros(self.chi, dtype=data.dtype), data] )[self.idxNewL]
        elif array_type in ['gcr','Gcr','GCR']:
            return backend.hstack( [backend.zeros([self.chi, 1], dtype=data.dtype), data] )[:, self.idxNewR]

    
    # Compact (de-align) given data
    def compact_data(self, selected: bool, array_type: str, data: np.ndarray, first_charge: int, last_charge: int, first_charge_r: int = None, last_charge_r: int = None):
        """
        Parameters
        ----------
        selected : bool
            Whether the data to be aligned is selected already
        array_type : str
            What is the type of the data array to be selected
            (from 'll','Ll','LL','cl','Cl','CL','lr','Lr','LR',
            'cr','Cr','CR','glc','Glc','GLC','gcr','Gcr','GCR','T','t')
        data : ndarray
            Wata array to be selected
        first_charge : int
            The beginning charge of selection
        last_charge : int
            The ending charge of selection
        first_charge_r : int (optional)
            The beginning charge of selection in the 1st dimension. Only use when selecting from 2D data.
        last_charge_r : int (optional)
            The ending charge of selection in the 1st dimension. Only use when selecting from 2D data.
        """
        if first_charge_r == None and last_charge_r == None:
            # Compacting along 1 dimension
            side = 'not_specified'
            if array_type in ['ll','Ll','LL','cl','Cl','CL','glc','Glc','GLC']:
                side = 'left'
                idxInv = self.idxInvL
                incNew = self.incNewL
            elif array_type in ['lr','Lr','LR','cr','Cr','CR','gcr','Gcr','GCR']:
                side = 'right'
                idxInv = self.idxInvR
                incNew = self.incNewR
            if side != 'not_specified':
                first_index, last_index = self.get_first_and_last_indices(first_charge, last_charge, side, False)
                idxInv = idxInv[first_index : last_index]
                if selected:
                    idxInv = idxInv - incNew[first_charge]
                if array_type in ['gcr','Gcr','GCR']:
                    return data[:, idxInv]
                else:
                    return data[idxInv]
            else:
                raise ValueError("Not a valid de-align select option.")

        elif first_charge_r != None and last_charge_r != None:
            # Compacting both dimensions (Only for  T)
            assert array_type in ['T', 't'], "If two sets of first and last charges are passed, the type must be T."
            first_compact_index_l, last_compact_index_l = self.get_first_and_last_indices(first_charge, last_charge, 'left', False)
            first_compact_index_r, last_compact_index_r = self.get_first_and_last_indices(first_charge_r, last_charge_r, 'right', False)
            idxInvl = self.idxInvL[first_compact_index_l : last_compact_index_l]
            idxInvr = self.idxInvR[first_compact_index_r : last_compact_index_r]
            if selected:
                first_aligned_index_l, _ = self.get_first_and_last_indices(first_charge, last_charge, 'left', True)
                first_aligned_index_r, _ = self.get_first_and_last_indices(first_charge_r, last_charge_r, 'right', True)
                idxInvl = idxInvl - first_aligned_index_l
                idxInvr = idxInvr - first_aligned_index_r
            return data[idxInvl][:, idxInvr]

        else:
            raise ValueError("Only Neither or Both first_charge_r and last_charge_r must be specified.")        


    # Select data based on smallest and largest charges
    def select_data(self, aligned: bool, array_type: str, data: np.ndarray, first_charge: int, last_charge: int, first_charge_1: int = None, last_charge_1: int = None):
        """
        Parameters
        ----------
        aligned : bool
            Whether the data to be selected is aligned already
        array_type : str
            What is the type of the data array to be selected
            (from 'll','Ll','LL','cl','Cl','CL', 'cc','Cc','CC','lr','Lr','LR',
            'cr','Cr','CR','glc','Glc','GLC','gcr','Gcr','GCR','T','t')
        data : ndarray
            Wata array to be selected
        first_charge : int
            The beginning charge of selection
        last_charge : int
            The ending charge of selection
        first_charge_1 : int (optional)
            The beginning charge of selection in the 1st dimension. Only use when selecting from 2D data.
        last_charge_1 : int (optional)
            The ending charge of selection in the 1st dimension. Only use when selecting from 2D data.
        """
        if first_charge_1 == None and last_charge_1 == None:
            # Selecting 1d data
            side = 'not_specified'
            if array_type in ['ll','Ll','LL','cl','Cl','CL']:
                side = 'left'
            if array_type in ['lc','Lc','LC','cc','Cc','CC']:
                side = 'center'
            elif array_type in ['lr','Lr','LR','cr','Cr','CR']:
                side = 'right'
            if side != 'not_specified':
                first_index, last_index = self.get_first_and_last_indices(first_charge, last_charge, side, aligned)
                return data[first_index : last_index]
            else:
                raise ValueError("Not a valid data select option.")

        elif first_charge_1 != None and last_charge_1 != None:
            # Selecting 2d data
            side0 = 'not_specified'
            if array_type in ['glc','Glc','GLC']:
                side0, side1 = 'left', 'center'
            elif array_type in ['gcr','Gcr','GCR']:
                side0, side1 = 'center', 'right'
            elif array_type in ['T', 't']:
                side0, side1 = 'left', 'right'
            if side1 != 'not_specified':
                first_index_0, last_index_0 = self.get_first_and_last_indices(first_charge, last_charge, side0, aligned)
                first_index_1, last_index_1 = self.get_first_and_last_indices(first_charge_1, last_charge_1, side1, aligned)
                return data[first_index_0 : last_index_0, first_index_1 : last_index_1]
            else:
                raise ValueError("Not a valid data select option.")

        else:
            raise ValueError("Only Neither or Both first_charge_r and last_charge_r must be specified.")    
   

    # Obtain the indices of a sorted array where the entries increases. Used for charges.
    @staticmethod
    def get_charge_beginning_index(d: int, charges: np.ndarray):

        size = charges.shape[0]
        inc = - np.ones(d + 1, dtype=charges.dtype) # Includes terminating charge value d.
        inc[d] = size # Initialize terminating charge increment index as the full size

        for charge_idx in range(size - 1):

            charge = charges[charge_idx].item()
            next_charge = charges[charge_idx + 1].item()

            if (charge_idx == 0 and charge != d):
                inc[charge] = 0

            if (charge != next_charge):
                inc[next_charge] = charge_idx + 1
        

        return inc

    # Given the sorted charge indices and the increment indices, remap them to align the charge boundaries to indices of multiples of 8
    @staticmethod
    def align_info(chi: int, inc: np.ndarray):

        d = inc.shape[0] - 1
        linear_indices = np.arange(chi, dtype=inc.dtype)

        # Finding the index offset needed for each charge value
        Offset = int(0)
        incidx = int(-1)
        c = int(0)
        old_c = c
        Offsets = np.zeros(d + 1, dtype=inc.dtype)
        incNew = - np.ones(d + 1, dtype=inc.dtype)
        
        while (c <= d):

            while (c <= d and incidx == -1):

                incidx = inc[c]

                if incidx != -1:
                    old_c = c

                c += 1

            if (incidx == -1):
                old_c = c - 1
                incidx = chi
            
            OffsetAdd = (8 - (incidx + Offset) % 8) % 8
            incidx = -1
            Offset += OffsetAdd
            Offsets[old_c] = Offset

            if (inc[old_c] != -1):
                incNew[old_c] = inc[old_c] + Offset
                
        # Dimension of the new array to store the data
        sizeNew = int(((chi + Offsets[old_c] + 7) // 8) * 8)

        # Create a new array to hold the data
        cNew = np.zeros(sizeNew, dtype=inc.dtype)
        idxNew = np.zeros(sizeNew, dtype=inc.dtype)
        idxInv = np.zeros(chi, dtype=inc.dtype)

        # Fill in a new charge array
        c = 0
        old_c = c
        incidx = 0
        for i in range(sizeNew):

            if (i == incidx):

                old_c = c
                c += 1
                incidx = -1

                while (c < d and incidx <= 0):

                    incidx = incNew[c]

                    if (incidx == 0):
                        old_c = c

                    c += 1

                c -= 1

            cNew[i] = old_c

        # Fill in index mapping array
        c = int(0)
        old_c = c
        incidx = int(0)
        for i in range(chi):

            if (i == incidx):

                old_c = c
                c += 1
                incidx = -1

                while (c < d and incidx <= 0):

                    incidx = inc[c]

                    if (incidx == 0):

                        old_c = c
        
                    c += 1

                c -= 1
                
            idxNew[i + Offsets[old_c]] = linear_indices[i] + 1; # 0th entry is reserved for 0 values
            idxInv[linear_indices[i]] = i + Offsets[old_c]


        return sizeNew, idxNew, idxInv, incNew, cNew


    # Finding index boundaries to select from the full indices
    def get_first_and_last_indices(self, first_charge: int, last_charge: int, side: str, aligned: bool) -> tuple[int, int]:

        if aligned:
            if side == 'left':
                inc = self.incNewL
            elif side == 'center':
                inc = self.incC
            elif side == 'right':
                inc = self.incNewR
        else:
            if side == 'left':
                inc = self.incL
            elif side == 'center':
                inc = self.incC
            elif side == 'right':
                inc = self.incR

        first_index = -1
        while first_index == -1:
            first_index = inc[first_charge]
            first_charge += 1

        if last_charge == inc.shape[0] - 1:
            # Special case of trying to select all data
            last_index = None
        else:
            last_index = -1
            while last_index == -1:
                last_index = inc[last_charge + 1]
                last_charge += 1

        return first_index, last_index





# Test if sort and alignment works
if __name__ == '__main__':

    np.set_printoptions(precision=1)

    chi = 15
    d = 6
    CL = np.random.randint(0, high=d-1, size=chi)
    for i in range(chi):
        if CL[i] == 4:
            CL[i] = 3
    CL = np.sort(CL)
    print('charges: ', CL)
    CC = np.random.randint(0, high=d-1, size=chi)
    for i in range(chi):
        if CC[i] == 3:
            CC[i] = 4
    CC = np.sort(CC)
    print('charges: ', CC)
    CR = np.random.randint(1, high=d-1, size=chi)
    for i in range(chi):
        if CR[i] == 2:
            CR[i] = 3
    CR = np.sort(CR)
    print('charges: ', CR)

    aligner = Aligner(d, chi, CL, CC, CR)

    sizeNewL, idxNewL, idxInvL, incNewL, cNewL = aligner.sizeNewL, aligner.idxNewL, aligner.idxInvL, aligner.incNewL, aligner.cNewL
    print('sizeNewL', sizeNewL)
    print('idxNewL', idxNewL)
    print('idxInvL', idxInvL)
    print('incNewL', incNewL)
    print('cNewL', cNewL.reshape(-1, 8))
    print(np.append(np.nan, CL)[idxNewL].reshape(-1, 8))

    sizeNewR, idxNewR, idxInvR, incNewR, cNewR = aligner.sizeNewR, aligner.idxNewR, aligner.idxInvR, aligner.incNewR, aligner.cNewR
    print('sizeNewR', sizeNewR)
    print('idxNewR', idxNewR)
    print('idxInvR', idxInvR)
    print('incNewR', incNewR)
    print('cNewR', cNewR.reshape(-1, 8))
    print(np.append(np.nan, CR)[idxNewR].reshape(-1, 8))

    Glc = np.random.rand(chi, chi)
    Gcr = np.random.rand(chi, chi)
    reference_T = np.matmul(Glc, Gcr)
    selected_reference_T = aligner.select_data(False, 'T', reference_T, 1,3,0,3)
    print('Reference T: ', reference_T)
    print('selected referece T: ', selected_reference_T)
    Glc = aligner.align_data('Glc', Glc)
    Gcr = aligner.align_data('Gcr', Gcr)
    print('Glc ', np.array(Glc != 0, dtype = int).reshape(sizeNewL, chi))
    print('Gcr: ', np.array(Gcr != 0, dtype = int).reshape(chi, sizeNewR))
    aligned_T = np.matmul(Glc, Gcr)
    print('Aligned T: ', np.array(aligned_T != 0, dtype = int).reshape(sizeNewL, sizeNewR))
    T = aligner.compact_data('T', aligned_T, 0, d-1, 0, d-1)
    print('Compact T: ', np.array(T != 0, dtype = int).reshape(chi, chi))
    print('Same? ', np.allclose(T, reference_T))
    selected_T = aligner.select_data(False, 'T', T, 1,3,0,3)
    print('selected T: ', selected_T)
    print('Same? ', np.allclose(selected_reference_T, selected_T))
    selected_aligned_T = aligner.select_data(True, 'T', aligned_T, 1,3,0,3)
    print('selected aligned T: ', selected_aligned_T)
    compact_selected_T = aligner.compact_data('T', aligned_T, 1,3,0,3)
    print('compact selected T: ', compact_selected_T)
    print('Same? ', np.allclose(selected_reference_T, compact_selected_T))

    # for i in range(1, 20):
    #     size = 50 * i
    #     print("size: ", size)
    #     start = time.time()
    #     results = np.linalg.svd(np.random.rand(10, size, size))
    #     stop = time.time()
    #     print(stop - start)
    #     start = time.time()
    #     results = np.linalg.svd(np.random.rand(size, size))
    #     stop = time.time()
    #     print((stop - start) * 10)
    #     start = time.time()
    #     results = svdr(np.random.rand(10, 30 * size, 30 * size), size)
    #     stop = time.time()
    #     print(stop - start)
