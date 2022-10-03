import numpy as np
import cupy as cp
import time


class Data(object):

    def __init__(self, array_type: str, aligned: bool, data, idx_select: list):
        if array_type in ['ll','Ll','LL','lc','Lc','LC','lr','Lr','LR']:
            assert data.ndim == len(idx_select), "idx_select length does not match data dimension."
            assert data.ndim == 1, 'Type and dim not compatible.'
        elif array_type in ['glc','Glc','GLC','gcr','Gcr','GCR','T','t']:
            assert data.ndim == len(idx_select), "idx_select length does not match data dimension."
            assert data.ndim == 2, 'Type and dim not compatible.'
        elif array_type in ['cl','cc','cr','CL','CC','CR']:
            assert data.ndim == 2, 'Charges must have 2 dims'
            assert data.shape[1] == 2, 'Charges must have 2nd dim of size 2'
            assert len(idx_select) == 1, 'Charges must have one starting index'
        else:
            raise ValueError("Not a valid array type option.")

        self.ndim = data.ndim
        self.array_type = array_type
        self.aligned = aligned
        self.data = data
        self.idx_select = idx_select

    def cupy(self):
        self.data = cp.array(self.data)

    def numpy(self):
        if type(self.data) is cp._core.core.ndarray:
           self.data = cp.asnumpy(self.data)

    def clone(self):
        return Data(self.array_type, self.aligned, self.data, self.idx_select)


class Aligner(object):

    def __init__(self, d: int, CL: np.ndarray, CC: np.ndarray, CR: np.ndarray) -> None:
        self.d = d
        self.chi = CL.shape[0]

        self.CL = self.make_data_obj('CL', True, CL, [0])
        self.CC = self.make_data_obj('CC', True, CC, [0])
        self.CR = self.make_data_obj('CR', True, CR, [0])

        self.align_info_time = 0
        self.index_time = 0

        start = time.time()
        self.incL_0, self.incL_1 = self.get_charge_beginning_index(d, CL)
        self.index_time += time.time() - start
        start = time.time()
        self.sizeNewL, self.idxNewL, cNewL = self.align_info(self.incL_0, self.incL_1)
        self.idxRealNewL = np.where(self.idxNewL > 0)[0]
        self.cNewL = self.make_data_obj('CL', True, cNewL, [0])
        self.align_info_time += time.time() - start

        start = time.time()
        _, incC_1 = self.get_charge_beginning_index(d, CC, absolute_index=True)
        self.change_charges_C = np.where(incC_1 != -1)
        self.change_idx_C = np.array(incC_1[self.change_charges_C], dtype='int32')
        self.change_charges_C = np.array(self.change_charges_C, dtype='int32')
        self.index_time += time.time() - start

        start = time.time()
        self.incR_0, self.incR_1 = self.get_charge_beginning_index(d, CR)
        self.index_time += time.time() - start
        start = time.time()
        self.sizeNewR, self.idxNewR, cNewR = self.align_info(self.incR_0, self.incR_1)
        self.idxRealNewR = np.where(self.idxNewR > 0)[0]
        self.cNewR = self.make_data_obj('CR', True, cNewR, [0])
        self.align_info_time += time.time() - start

    # Align given data
    def align_data(self, data_obj: Data):
        array_type = data_obj.array_type
        data = data_obj.data
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
            data_obj.data = backend.append(np.zeros(1, dtype=data.dtype), data)[self.idxNewL]
        elif array_type in ['lr','Lr','LR']:
            data_obj.data = backend.append(np.zeros(1, dtype=data.dtype), data)[self.idxNewR]
        elif array_type in ['glc','Glc','GLC']:
            data_obj.data = backend.vstack( [backend.zeros(self.chi, dtype=data.dtype), data] )[self.idxNewL]
        elif array_type in ['gcr','Gcr','GCR']:
            data_obj.data = backend.hstack( [backend.zeros([self.chi, 1], dtype=data.dtype), data] )[:, self.idxNewR]

        return data_obj

    
    # Compact (de-align) given data
    def compact_data(self, data_obj: Data):

        array_type = data_obj.array_type
        data = data_obj.data
        idx_select = data_obj.idx_select

        if array_type in ['T', 't']:
            idx_select_l = idx_select[0]
            idx_select_r = idx_select[1]
            _, indices_l, _ = np.intersect1d(idx_select_l, self.idxRealNewL, return_indices=True)
            _, indices_r, _ = np.intersect1d(idx_select_r, self.idxRealNewR, return_indices=True)
            return data[indices_l][:, indices_r]
        else:
            # Compacting along 1 dimension
            idx_select = idx_select[0]
            if array_type in ['ll','Ll','LL','cl','Cl','CL','glc','Glc','GLC']:
                idxRealNew = self.idxRealNewL
            elif array_type in ['lr','Lr','LR','cr','Cr','CR','gcr','Gcr','GCR']:
                idxRealNew = self.idxRealNewR
            else:
                raise ValueError("Not a valid de-align select option.")
            _, indices, _ = np.intersect1d(idx_select, idxRealNew, return_indices=True)
            if array_type in ['gcr','Gcr','GCR']:
                return data[:, indices]
            else:
                return data[indices]

    # Select data based on smallest and largest charges
    def select_data(self, data_obj: Data, first_charge_0: int, last_charge_0: int, first_charge_1: int, last_charge_1: int, first_charge_another_0: int = None, last_charge_another_0: int = None, first_charge_another_1: int = None, last_charge_another_1: int = None):

        aligned = data_obj.aligned
        array_type = data_obj.array_type

        if first_charge_another_0 == None and last_charge_another_0 == None and first_charge_another_1 == None and last_charge_another_1 == None:
            # Selecting 1d data
            side = 'not_specified'
            if array_type in ['ll','Ll','LL','cl','Cl','CL']:
                side = 'left'
            if array_type in ['lc','Lc','LC','cc','Cc','CC']:
                side = 'center'
            elif array_type in ['lr','Lr','LR','cr','Cr','CR']:
                side = 'right'
            if side != 'not_specified':
                idx_select = self.get_indices(first_charge_0, last_charge_0, first_charge_1, last_charge_1, side, aligned)
                return Data(array_type, aligned, data_obj.data[idx_select], [idx_select])
            else:
                raise ValueError("Not a valid data select option.")

        elif first_charge_another_0 != None and last_charge_another_0 != None and first_charge_another_1 != None and last_charge_another_1 != None:
            # Selecting 2d data
            side0 = 'not_specified'
            if array_type in ['glc','Glc','GLC']:
                side0, side1 = 'left', 'center'
            elif array_type in ['gcr','Gcr','GCR']:
                side0, side1 = 'center', 'right'
            elif array_type in ['T', 't']:
                side0, side1 = 'left', 'right'
            if side1 != 'not_specified':
                idx_select_0 = self.get_indices(first_charge_0, last_charge_0, first_charge_1, last_charge_1, side0, aligned)
                idx_select_1 = self.get_indices(first_charge_another_0, last_charge_another_0, first_charge_another_1, last_charge_another_1, side1, aligned)
                return Data(array_type, aligned, data_obj.data[idx_select_0][:, idx_select_1], [idx_select_0, idx_select_1])
            else:
                raise ValueError("Not a valid data select option.")

        else:
            raise ValueError("Only None or all first_charge_left/right_0/1 must be specified.")


    # Select data based on smallest and largest charges
    def get_select_index(self, data_obj: Data, first_charge_0: int, last_charge_0: int, first_charge_1: int, last_charge_1: int, first_charge_another_0: int = None, last_charge_another_0: int = None, first_charge_another_1: int = None, last_charge_another_1: int = None):

        aligned = data_obj.aligned
        array_type = data_obj.array_type

        if first_charge_another_0 == None and last_charge_another_0 == None and first_charge_another_1 == None and last_charge_another_1 == None:
            # Selecting 1d data
            side = 'not_specified'
            if array_type in ['ll','Ll','LL','cl','Cl','CL']:
                side = 'left'
            if array_type in ['lc','Lc','LC','cc','Cc','CC']:
                side = 'center'
            elif array_type in ['lr','Lr','LR','cr','Cr','CR']:
                side = 'right'
            if side != 'not_specified':
                idx_select = self.get_indices(first_charge_0, last_charge_0, first_charge_1, last_charge_1, side, aligned)
                return idx_select
            else:
                raise ValueError("Not a valid data select option.")

        elif first_charge_another_0 != None and last_charge_another_0 != None and first_charge_another_1 != None and last_charge_another_1 != None:
            # Selecting 2d data
            side0 = 'not_specified'
            if array_type in ['glc','Glc','GLC']:
                side0, side1 = 'left', 'center'
            elif array_type in ['gcr','Gcr','GCR']:
                side0, side1 = 'center', 'right'
            elif array_type in ['T', 't']:
                side0, side1 = 'left', 'right'
            if side1 != 'not_specified':
                idx_select_0 = self.get_indices(first_charge_0, last_charge_0, first_charge_1, last_charge_1, side0, aligned)
                idx_select_1 = self.get_indices(first_charge_another_0, last_charge_another_0, first_charge_another_1, last_charge_another_1, side1, aligned)
                return idx_select_0, idx_select_1
            else:
                raise ValueError("Not a valid data select option.")

        else:
            raise ValueError("Only None or all first_charge_left/right_0/1 must be specified.")   
   

    # Obtain the indices of a sorted array where the entries increases. Used for charges.
    @staticmethod
    def get_charge_beginning_index(d: int, charges: np.ndarray, absolute_index=False):

        charges_0 = charges[:, 0]
        size_0 = charges_0.shape[0]
        inc_0 = - np.ones(d + 1, dtype='int32') # Includes terminating charge value d.
        inc_1 = - np.ones([d + 1, d + 1], dtype='int32')

        for idx_0 in range(size_0):

            charge_0 = charges_0[idx_0].item()
            if idx_0 == size_0 - 1:
                next_charge_0 = d
            else:
                next_charge_0 = charges_0[idx_0 + 1].item()

            if idx_0 == 0 and charge_0 != d:
                inc_0[charge_0] = 0

            if charge_0 != next_charge_0:
                inc_0[next_charge_0] = idx_0 + 1
                charges_1 = charges[ inc_0[charge_0] : inc_0[next_charge_0] , 1]
                size_1 = charges_1.shape[0]

                for idx_1 in range(size_1):

                    charge_1 = charges_1[idx_1].item()
                    if idx_1 == size_1 - 1:
                        next_charge_1 = d
                    else:
                        next_charge_1 = charges_1[idx_1 + 1].item()

                    if idx_1 == 0 and charge_1 != d:
                        inc_1[charge_0, charge_1] = 0
                        if absolute_index:
                            inc_1[charge_0, charge_1] += inc_0[charge_0]

                    if charge_1 != next_charge_1:
                        inc_1[charge_0, next_charge_1] = idx_1 + 1
                        if absolute_index:
                            inc_1[charge_0, next_charge_1] += inc_0[charge_0]

        if absolute_index:
            inc_1 = inc_1[:, :-1]
        
        return inc_0, inc_1

    # Given the sorted charge indices and the increment indices, remap them to align the charge boundaries to indices of multiples of 8
    @staticmethod
    def align_info(inc_0: np.ndarray, inc_1: np.ndarray):

        d = inc_0.shape[0] - 1

        sizeNew = 0
        idxNew = np.array([], 'int32')
        incNew1 = - np.ones(d + 1, 'int32')
        incNew2 = - np.ones([d + 1, d + 1], 'int32')
        cNew = np.array([], 'int32').reshape(0, 2)

        idxOffset = 0

        for c_0 in range(d):

            inc2 = inc_1[c_0]
            size2 = inc2[-1]
            if size2 == -1:
                continue

            incNew1[c_0] = sizeNew
            
            linear_indices = np.arange(size2, dtype=inc2.dtype)

            # Finding the index offset needed for each charge value
            Offset = int(0)
            incidx = int(-1)
            c = int(0)
            old_c = c
            Offsets = np.zeros(d + 1, dtype=inc2.dtype)
            incnew2 = - np.ones(d + 1, dtype=inc2.dtype)
            
            while (c <= d):

                while (c <= d and incidx == -1):

                    incidx = inc2[c]

                    if incidx != -1:
                        old_c = c

                    c += 1

                if (incidx == -1):
                    old_c = c - 1
                    incidx = size2
                
                OffsetAdd = (8 - (incidx + Offset) % 8) % 8
                incidx = -1
                Offset += OffsetAdd
                Offsets[old_c] = Offset

                if (inc2[old_c] != -1):
                    incnew2[old_c] = inc2[old_c] + Offset
                    
            # Dimension of the new array to store the data
            sizeNew2 = int(((size2 + Offsets[old_c] + 7) // 8) * 8)
            # Create a new array to hold the data
            cNew2 = np.zeros([sizeNew2, 2], dtype=inc2.dtype)
            cNew2[:, 0] = c_0
            idxNew2 = np.zeros(sizeNew2, dtype=inc2.dtype)

            # Fill in a new charge array
            c = 0
            old_c = c
            incidx = 0
            for i in range(sizeNew2):

                if (i == incidx):

                    old_c = c
                    c += 1
                    incidx = -1

                    while (c < d and incidx <= 0):

                        incidx = incnew2[c]

                        if (incidx == 0):
                            old_c = c

                        c += 1

                    c -= 1

                cNew2[i, 1] = old_c

            # Fill in index mapping array
            c = int(0)
            old_c = c
            incidx = int(0)
            for i in range(size2):

                if (i == incidx):

                    old_c = c
                    c += 1
                    incidx = -1

                    while (c < d and incidx <= 0):

                        incidx = inc2[c]

                        if (incidx == 0):

                            old_c = c
            
                        c += 1

                    c -= 1
                    
                idxNew2[i + Offsets[old_c]] = linear_indices[i] + 1 + idxOffset # 0th entry is reserved for 0 values

            idxNew = np.append(idxNew, idxNew2)
            incnew2[np.where(incnew2 != -1)] += sizeNew
            incNew2[c_0] = incnew2
            cNew = np.append(cNew, cNew2, axis=0)
            sizeNew += sizeNew2
            idxOffset += size2

        incNew1[d] = sizeNew

        return sizeNew, idxNew, cNew

    @staticmethod
    def make_data_obj(array_type: str, aligned: bool, data, idx_select: list):
        return Data(array_type, aligned, data, idx_select)

    @staticmethod
    def to_cupy(data_obj: Data):
        d_data_obj = data_obj.clone()
        d_data_obj.cupy()
        return d_data_obj

    # Finding index boundaries to select from the full indices
    def get_indices(self, first_charge_0: int, last_charge_0: int, first_charge_1: int, last_charge_1: int, side: str, aligned: bool):# -> tuple[int, int]:

        if aligned:
            if side == 'left':
                charges = self.cNewL.data
            elif side == 'center':
                charges = self.CC.data
            elif side == 'right':
                charges = self.cNewR.data
        else:
            if side == 'left':
                charges = self.CL.data
            elif side == 'center':
                charges = self.CC.data
            elif side == 'right':
                charges = self.CR.data

        idx_0 = np.intersect1d(np.where(first_charge_0 <= charges[:, 0])[0], np.where(charges[:,0] <= last_charge_0)[0])
        idx_1 = np.intersect1d(np.where(first_charge_1 <= charges[:, 1])[0], np.where(charges[:,1] <= last_charge_1)[0])
        indices = np.intersect1d(idx_0, idx_1)

        return indices





# Test if sort and alignment works
if __name__ == '__main__':

    np.set_printoptions(precision=1)

    chi = 8**2
    d = 6
    CL = np.random.randint(0, high=d-1, size=[chi, 2])
    for i in range(chi):
        if CL[i,0] == 4:
            CL[i,0] = 3
    CL = CL[np.lexsort((CL[:, 1], CL[:, 0]))]
    print('charges: ', CL)
    CC = np.random.randint(0, high=d-1, size=[chi, 2])
    for i in range(chi):
        if CC[i,0] == 3:
            CC[i,0] = 4
    CC = CC[np.lexsort((CC[:, 1], CC[:, 0]))]
    print('charges: ', CC)
    CR = np.random.randint(1, high=d-1, size=[chi, 2])
    for i in range(chi):
        if CR[i,1] == 2:
            CR[i,1] = 3
    CR = CR[np.lexsort((CR[:, 1], CR[:, 0]))]
    print('charges: ', CR)

    aligner = Aligner(d, chi, CL, CC, CR)

    sizeNewL, idxNewL, idxInvL, incNewL1, cNewL = aligner.sizeNewL, aligner.idxNewL, aligner.idxInvL, aligner.incNewL1, aligner.cNewL
    print('sizeNewL', sizeNewL)
    print('idxNewL', idxNewL.reshape(-1, 8))
    print('idxInvL', idxInvL)
    print('incNewL1', incNewL1)
    print('cNewL', cNewL.data[:, 1].reshape(-1, 8))
    print(np.append(np.nan, CL[:, 1])[idxNewL].reshape(-1, 8))

    sizeNewR, idxNewR, idxInvR, incNewR1, cNewR = aligner.sizeNewR, aligner.idxNewR, aligner.idxInvR, aligner.incNewR1, aligner.cNewR
    print('sizeNewR', sizeNewR)
    print('idxNewR', idxNewR.reshape(-1, 8))
    print('idxInvR', idxInvR)
    print('incNewR1', incNewR1)
    print('cNewR', cNewR.data[:, 1].reshape(-1, 8))
    print(np.append(np.nan, CR[:, 1])[idxNewR].reshape(-1, 8))

    Glc = np.random.rand(chi, chi)
    Gcr = np.random.rand(chi, chi)
    reference_T_obj = aligner.make_data_obj('T', False, np.matmul(Glc, Gcr), [0, 0])
    selected_reference_T_obj = aligner.select_data(reference_T_obj, 1,3,0,2,0,3,0,2)
    selected_reference_T_obj.data = 1
    print(reference_T_obj.data)
    quit()
    #print('Reference T: ', reference_T_obj.data)
    #print('selected referece T: ', selected_reference_T)
    Glc = aligner.align_data('Glc', Glc)
    Gcr = aligner.align_data('Gcr', Gcr)
    print('Glc ', np.array(Glc != 0, dtype = int).reshape(sizeNewL, chi))
    print('Gcr: ', np.array(Gcr != 0, dtype = int).reshape(chi, sizeNewR))
    aligned_T_obj = aligner.make_data_obj('T', True, np.matmul(Glc.data, Gcr.data), [0, 0])
    #print('Aligned T: ', np.array(aligned_T != 0, dtype = int).reshape(sizeNewL, sizeNewR))
    T = aligner.compact_data(aligned_T_obj, 1,3,0,2,0,3,0,2)
    print('Aligned T: ', np.array(aligned_T_obj.data[0].reshape(-1, 8) != 0, dtype = int))
    print('Aligned Compact T: ', np.array(T != 0, dtype = int))
    print('Same? ', np.allclose(T, selected_reference_T_obj.data))

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
