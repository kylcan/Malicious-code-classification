#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   features.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author           @Version    @Desciption
------------      -------           --------    -----------
2021-10-04 17:17   PowerLZY&yuan_mes  1.0       归档提取函数
"""


import re
import numpy as np
from sklearn.feature_extraction import FeatureHasher



class FeatureType(object):
    """ Base class from which each feature type may inherit. """

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez):
        """ Generate a JSON-able representation of the file. """
        raise NotImplementedError

    def process_raw_features(self, raw_obj):
        """ Generate a feature vector from the raw features. """
        raise NotImplementedError

    def feature_vector(self, bytez):
        """ Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. """
        return self.process_raw_features(self.raw_features(bytez))


# Format-agnostic features (PE files)
class ByteHistogram(FeatureType):
    """ Byte histogram (count + non-normalized) over the entire binary file. """

    name = 'histogram'
    dim = 256

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


class ByteEntropyHistogram(FeatureType):
    """ 2d byte/entropy histogram, which roughly approximates the joint probability of byte value and local entropy. """

    name = 'byteentropy'
    dim = 256

    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.step = step
        self.window = window

    def _entropy_bin_counts(self, block):
        # Coarse histogram, 16 bytes per bin
        # 16-bin histogram
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / self.window
        # Filter non-zero elements
        wh = np.where(c)[0]
        # "* 2" b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4bits)
        H = np.sum(-p[wh] * np.log2(p[wh])) * 2
        # Up to 16 bins (max entropy is 8 bits)
        Hbin = int(H * 2)
        # Handle entropy = 8.0 bits
        if Hbin == 16:
            Hbin = 15
        return Hbin, c

    def raw_features(self, bytez):
        output = np.zeros((16, 16), dtype=int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # Strided trick
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            # From the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c
        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized


# String-like features (PE & asm files)
class StringExtractor(FeatureType):
    """ Extracts strings from raw byte stream of PE or asm file. """

    name = 'strings'
    name_tfidf = 'words'
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1  # 104

    def __init__(self):
        super(FeatureType, self).__init__()
        # All consecutive runs of printable string that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # Occurrences of the string 'C:\', not actually extracting the path.
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # Occurrences of 'http://' or 'https://', not actually extracting the URLs.
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # Occurrences of the string prefix 'HKEY_', not actually extracting registry names.
        self._registry = re.compile(b'HKEY_')
        # Crude evidence of an MZ header (PE dropper or bubbled executable) somewhere in the byte stream
        self._mz = re.compile(b'MZ')
        # all words which can read
        self._words = re.compile(b"[a-zA-Z]+")

    def tfidf_features(self, bytez):
        """ Extracts a list of readable strings for tf-idf """
        list_ = []
        list2 = []
        words = []
        for line in bytez:
            raw_words = re.findall('[a-zA-Z]+', line)
            words_space = ' '.join(w for w in raw_words if 4 < len(w) < 20)
            list_.append(words_space)
        for item in list_:  # 第二轮清洗，过滤掉小于3的字符串
            if len(item) > 3:
                list2.append(item)
        for item in list2:  # 第三轮清洗,对过长的字符串进行拆分
            if len(item) > 20:
                for text in item.split():
                    if (('a' in text) or ('e' in text) or ('i' in text) or ('o' in text) or ('u' in text) or (
                            'A' in text) or ('E' in text) or ('I' in text) or ('O' in text) or ('U' in text)):
                        if ('abcdef' not in text) and ('aaaaaa' not in text) and ('<init>' not in text):
                            words.append(text)

        return words

    def raw_features(self, bytez):
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # Statistics about strings
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # Map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            # Histogram count
            c = np.bincount(as_shifted_string, minlength=96)
            # Distribution of characters in printable strings (entropy)
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            csum = 0
            H = 0
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)


# Parsed features (extracted from asm file to supplement PE features)
class SectionInfo(FeatureType):
    """ Information about section names, sizes and certain special sections. Uses hashing trick to summarize all this
    section info into a feature vector. """

    name = 'section'
    dim = 10 + 50 + 50 + 50 + 50  # 210

    def __init__(self):
        super(FeatureType, self).__init__()
        # Beginning of the section
        self._section = re.compile(r'; Section (\d+)')
        # Segment names
        self._name = re.compile(r'\n(\S+)\s+segment')
        self._idata = re.compile(r'; (_idata)')
        # Section size in file and virtual size
        self._size = re.compile(r'Section size in file.*?[(]\s*?(\d+)')
        self._vsize = re.compile(r'Virtual size.*?[(]\s*?(\d+)')
        # Characteristics of section
        self._properties = re.compile(r'Flags\s+\w+:\s(.+)')
        self._entryperm = re.compile(r'Segment permissions:\s(\S+)')
        self._entrytype = re.compile(r'Segment type:\s+(.+)')

    def raw_features(self, bytez):
        # Collect infos by section order
        section_size = self._size.findall(bytez)
        section_size = [int(s) for s in section_size]
        virtual_size = self._vsize.findall(bytez)
        virtual_size = [int(v) for v in virtual_size]
        properties = [p.split() for p in self._properties.findall(bytez)]
        # Restrict the scope of sections
        section_id = [int(s) for s in self._section.findall(bytez)]
        section_pos = [pos.span()[0] for pos in self._section.finditer(bytez)]
        # Infos of first executable section are complete (but need to be discarded)
        if 1 in section_id:
            section_pos.pop(0)
            section_size.pop(0)
            virtual_size.pop(0)
            properties.pop(0)
        section_pos.append(len(bytez))
        section_name = []
        for i in range(1, len(section_pos)):
            # Integrate segments into sections
            segment_name = self._name.findall(bytez, section_pos[i - 1], section_pos[i])
            is_idata = self._idata.findall(bytez, section_pos[i - 1], section_pos[i])
            if len(is_idata) != 0:
                segment_name.extend(is_idata)
            section_name.append(' '.join(segment_name))
        # Entry point, that is, the first executable section
        entry_sec = self._name.findall(bytez, 0, section_pos[0])
        is_idata = self._idata.findall(bytez, 0, section_pos[0])
        entry_type = self._entrytype.findall(bytez, 0, section_pos[0])[0]
        if entry_type == 'Externs':
            entry_props = []
        else:
            entry_type = entry_type.split()[-1].capitalize().replace('Code', 'Text')
            entry_props = [entry_type]
        if len(entry_sec) == 0:
            if len(is_idata) != 0:
                entry_section = is_idata[0]
            else:
                entry_section = ''
        else:
            if len(is_idata) != 0:
                entry_sec.extend(is_idata)
            entry_section = ' '.join(entry_sec)
            entry_perm = self._entryperm.findall(bytez, 0, section_pos[0])
            perms = []
            for ep in entry_perm:
                ep_str = ep.replace('Read', 'Readable').replace('Write', 'Writable').replace('Execute', 'Executable')
                perms.extend(ep_str.split('/'))
            entry_perms = list(set(perms))
            entry_perms.sort(key=perms.index)
            entry_props.extend(entry_perms)
        raw_obj = {"entry": {'name': entry_section, 'props': entry_props},
                   "sections": [{'name': name, 'size': size, 'vsize': vsize, 'props': props}
                                for name, size, vsize, props in
                                zip(section_name, section_size, virtual_size, properties)]}
        return raw_obj

    def process_raw_features(self, raw_obj):
        entry = raw_obj['entry']
        sections = raw_obj['sections']
        # Split the section name into segment names
        segments = [entry['name']]
        segments.extend(s['name'] for s in sections)
        segments = ' '.join(segments).split()
        # Get the permission of RX/W of first executable section
        rx = 1 if 'Readable' in entry['props'] and 'Executable' in entry['props'] else 0
        w = 1 if 'Writable' in entry['props'] else 0
        general = [
            # Total number of sections
            len(sections) + 1,
            # Total number of segments
            len(segments),
            # number of sections with zero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ''),
            # Number of RX
            sum(1 for s in sections if 'Readable' in s['props'] and 'Executable' in s['props']) + rx,
            # Number of W
            sum(1 for s in sections if 'Writable' in s['props']) + w,
            # If debug section exists
            1 if '_debug' in segments else 0,
            # If relocation section exists
            1 if '_reloc' in segments else 0,
            # If resource section exists
            1 if '_rsrc' in segments else 0,
            # If thread local storage section exists
            1 if '_tls' in segments else 0
        ]
        # Gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([entry['name']]).toarray()[0]
        characteristics = entry['props'] + [p for s in sections for p in s['props']
                                            if s['name'].find(entry['name'].split()[0]) != -1]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
        return np.hstack([
            general, section_sizes_hashed, section_vsize_hashed, entry_name_hashed, characteristics_hashed
        ]).astype(np.float32)


class ImportsInfo(FeatureType):
    """ Information about imported libraries and functions from the import address table. """

    name = 'imports'
    dim = 256 + 1024 + 1  # 1281

    def __init__(self):
        super(FeatureType, self).__init__()
        # Imported libraries and functions
        self._libraries = re.compile(r'Imports from (.*?dll)')
        self._functions = re.compile(r'extrn (\w+)')

    def raw_features(self, bytez):
        imports = {}
        # Restrict the function area of libraries
        libraries = self._libraries.findall(bytez)
        if len(libraries) == 0:
            return imports
        library_pos = [pos.span()[0] for pos in self._libraries.finditer(bytez)]
        library_pos.append(len(bytez))
        for i in range(1, len(library_pos)):
            functions = self._functions.findall(bytez, library_pos[i - 1], library_pos[i])
            functions = [f.replace('__imp_', '') for f in functions]
            # Libraries can be duplicated in listing, extend instead of overwrite
            if libraries[i - 1] not in imports:
                imports[libraries[i - 1]] = functions
            else:
                imports[libraries[i - 1]].extend(functions)
        return imports

    def process_raw_features(self, raw_obj):
        # Unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]
        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]
        # Total number of import functions
        functions = []
        for func in raw_obj.values():
            functions.extend(func)
        imports_func_num = [len(functions)]
        return np.hstack([libraries_hashed, imports_hashed, imports_func_num]).astype(np.float32)


class ExportsInfo(FeatureType):
    """ Information about exported functions. """

    name = 'exports'
    dim = 128

    def __init__(self):
        super(FeatureType, self).__init__()
        # Exported functions
        self._functions = re.compile(r'Exported entry\s+\d.\s([a-zA-Z_]+)')

    def raw_features(self, bytez):
        functions = self._functions.findall(bytez)
        if len(functions) == 0:
            return []
        exports = list(set([f.lower() for f in functions]))
        return exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        #export_func_num = [len(raw_obj)]
        return exports_hashed.astype(np.float32)


class OpcodeInfo(FeatureType):
    '''
    Information about "intesting" Opcode N-gram from the asm file.
    '''

    name_tfidf = 'ins'

    # dim = 0

    def __init__(self):
        super(FeatureType, self).__init__()
        self.start_key_ = 'Segment type:	Pure code'
        self.end_key_ = "		ends"
        self.run_ = 0
        self.key_ = "		"
        self.evasion_key_ = ['dd', 'dq', 'db', 'dw', 'unicode', ';org', 'assume', 'align', ';', 'public']

        # 数据寄存器 EAX, EBX, ECX, EDX (Data Register)
        # 指针寄存器 EBP, ESP (Pointer Register)
        # 变址寄存器 ESI, EDI (Index Register)
        self.register_ = [
            'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
            'eax,', 'ebx,', 'ecx,', 'edx,', 'esi,', 'edi,', 'ebp,', 'esp,',
            'ah', 'al', 'bh', 'bl', 'ch', 'cl', 'dh', 'dl',
            'ah,', 'al,', 'bh,', 'bl,', 'ch,', 'cl,', 'dh,', 'dl,',
            'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
            'ax,', 'bx,', 'cx,', 'dx,', 'si,', 'di,', 'bp,', 'sp,',
        ]
        self.no_call_ = ['sub_', 'dword']

    def tfidf_features(self, bytez):
        ins_list = []
        run = 0

        for line in bytez:
            line = str(line)
            if self.start_key_ in line:
                run = 1
            if self.end_key_ in line:
                run = 0
            if run == 1:
                if line.startswith(self.key_):
                    line_xx = line.split()
                    if (len(line_xx) > 1) and (line_xx[0] not in self.evasion_key_):
                        op = line_xx[0]
                        if line_xx[0] == 'call':
                            if not line_xx[1].startswith('sub') and not line_xx[1].startswith('dword') \
                                    and not line_xx[1].startswith('unknown_'):
                                op = op + line_xx[1]

                        if line_xx[1] in self.register_:
                            op = op + line_xx[1].split(',')[0]

                        if line_xx[-2] == ';':
                            op = op + line_xx[-1]
                        ins_list.append(op)
                elif len(ins_list) > 0 and ins_list[-1] != ';':
                    ins_list.append(';')

        return ins_list

    def asm_to_txt(self, bytez):
        """ asm opcode + string 保存到文件 """
        opline = ''
        opline_list = []
        run = 0

        for line in bytez:
            line = str(line)
            if self.start_key_ in line:
                run = 1
            if self.end_key_ in line:
                run = 0
            if run == 1:
                if line.startswith(self.key_):
                    line_xx = line.split()
                    if (len(line_xx) > 1) and (line_xx[0] not in self.evasion_key_):
                        opline = opline + line_xx[0] + ' '

                        if line_xx[0] == 'call':  # 添加调用函数
                            if not line_xx[1].startswith('sub') and not line_xx[1].startswith('dword') \
                                    and not line_xx[1].startswith('unknown_'):
                                opline = opline + line_xx[1] + ' '


                        """
                         for xx in line_xx:  # 添加寄存器
                            for r in self.register_:
                                if r in xx:
                                    opline = opline + r.split(',')[0] + ' '
                                    break
                       
                        """
                        # 添加寄存器
                        for xx in line_xx:
                            if xx in self.register_:
                                opline = opline + xx.split(',')[0] + ' '

                        if line_xx[-2] == ';':
                            opline = opline + line_xx[-1] + ' '
                        #opline = opline + ' '
                    elif len(opline) > 0:
                        opline_list.append(opline)
                        opline = ''

        return opline_list


class StringVector(FeatureType):
    """ String vector generated from .asm file. """

    name = 'semantic'
    dim = 1024

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, model_words):
        model_wv = model_words[0]
        raw_words = model_words[1]
        vector_list = [model_wv[key] for key in raw_words if key in model_wv]
        vector_arr = np.array(vector_list)
        return vector_arr

    def process_raw_features(self, raw_obj):
        return np.mean(raw_obj, axis=0)

