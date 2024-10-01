# Patients:
## Structure of Patients:
(0, 'PatientID', 'TEXT', 0, None, 1)
(1, 'Name', 'TEXT', 0, None, 0)
(2, 'Gender', 'TEXT', 0, None, 0)
(3, 'Birthdate', 'DATE', 0, None, 0)
(4, 'FirstVisitDate', 'DATE', 0, None, 0)
(5, 'FirstVisitAge', 'INTEGER', 0, None, 0)
(6, 'ShortestRevisitInterval', 'INTEGER', 0, None, 0)
(7, 'LongestRevisitInterval', 'INTEGER', 0, None, 0)
(8, 'AverageRevisitInterval', 'REAL', 0, None, 0)
(9, 'VisitDuration', 'INTEGER', 0, None, 0)
(10, 'TotalVisits', 'INTEGER', 0, None, 0)
(11, 'LastVisitDate', 'DATE', 0, None, 0)
(12, 'CIndU', 'INT', 0, None, 0)
## Foreign keys of Patients:


## Indexes of Patients:
(0, 'sqlite_autoindex_Patients_1', 1, 'pk', 0)


# OutpatientNumbers:
## Structure of OutpatientNumbers:
(0, 'OutpatientNumber', 'INTEGER', 0, None, 1)
(1, 'PatientID', 'TEXT', 0, None, 0)
## Foreign keys of OutpatientNumbers:
(0, 0, 'Patients', 'PatientID', 'PatientID', 'NO ACTION', 'NO ACTION', 'NONE')


## Indexes of OutpatientNumbers:
(0, 'idx_outpatientnumber_patientid', 0, 'c', 0)


# PatientVisits:
## Structure of PatientVisits:
(0, 'VisitID', 'INTEGER', 0, None, 1)
(1, 'OutpatientNumber', 'INTEGER', 0, None, 0)
(2, 'PatientID', 'TEXT', 0, None, 0)
(3, 'VisitDate', 'DATE', 0, None, 0)
(4, 'VisitType', 'TEXT', 0, None, 0)
(5, 'Department', 'TEXT', 0, None, 0)
(6, 'Doctor', 'TEXT', 0, None, 0)
(7, 'Diagnosis', 'TEXT', 0, None, 0)
(8, 'Disease', 'TEXT', 0, None, 0)
## Foreign keys of PatientVisits:
(0, 0, 'OutpatientNumbers', 'OutpatientNumber', 'OutpatientNumber', 'NO ACTION', 'NO ACTION', 'NONE')
(1, 0, 'Patients', 'PatientID', 'PatientID', 'NO ACTION', 'NO ACTION', 'NONE')


## Indexes of PatientVisits:
(0, 'idx_patientvisits_outpatientnumbers', 0, 'c', 0)
(1, 'idx_patientvisits_patientid', 0, 'c', 0)


# ExaminationToVisit:
## Structure of ExaminationToVisit:
(0, 'PatientID', 'TEXT', 0, None, 0)
(1, 'ExaminationDate', 'DATE', 0, None, 0)
(2, 'ExaminationSetID', 'INTEGER', 0, None, 1)
(3, 'VisitID', 'INTEGER', 0, 'NULL', 0)
## Foreign keys of ExaminationToVisit:
(0, 0, 'PatientVisits', 'VisitID', 'VisitID', 'NO ACTION', 'NO ACTION', 'NONE')


## Indexes of ExaminationToVisit:
(0, 'idx_etv_visitid', 0, 'c', 0)
(1, 'idx_etv_patientid_examinationdate', 0, 'c', 0)


# sqlite_sequence:
## Structure of sqlite_sequence:
(0, 'name', '', 0, None, 0)
(1, 'seq', '', 0, None, 0)
## Foreign keys of sqlite_sequence:


## Indexes of sqlite_sequence:


# NewPatientExaminations:
## Structure of NewPatientExaminations:
(0, 'ExaminationID', 'INTEGER', 0, None, 1)
(1, 'OutpatientNumber', 'INTEGER', 0, None, 0)
(2, 'ExaminationDate', 'DATE', 0, None, 0)
(3, 'ExaminationItemID', 'TEXT', 0, None, 0)
(4, 'ExaminationItem', 'TEXT', 0, None, 0)
(5, 'Result', 'REAL', 0, None, 0)
(6, 'NormalRange', 'TEXT', 0, None, 0)
(7, 'Unit', 'TEXT', 0, None, 0)
(8, 'PatientID', 'TEXT', 0, None, 0)
(9, 'VisitID', 'INTEGER', 0, None, 0)
(10, 'ExaminationSetID', 'INTEGER', 0, None, 0)
## Foreign keys of NewPatientExaminations:
(0, 0, 'Patients', 'PatientID', 'PatientID', 'NO ACTION', 'NO ACTION', 'NONE')
(1, 0, 'ExaminationToVisit', 'ExaminationSetID', 'ExaminationSetID', 'NO ACTION', 'NO ACTION', 'NONE')
(2, 0, 'PatientVisits', 'VisitID', 'VisitID', 'NO ACTION', 'NO ACTION', 'NONE')
(3, 0, 'ExaminationItems', 'ExaminationItemID', 'ExaminationItemID', 'NO ACTION', 'NO ACTION', 'NONE')
(4, 0, 'OutpatientNumbers', 'OutpatientNumber', 'OutpatientNumber', 'NO ACTION', 'NO ACTION', 'NONE')


## Indexes of NewPatientExaminations:
(0, 'idx_newpatientexaminations_examinationitemid', 0, 'c', 0)
(1, 'idx_newpatientexaminations_examinationsetid', 0, 'c', 0)
(2, 'idx_newwpatientexaminations_patientid', 0, 'c', 0)
(3, 'idx_newnpatientexaminations_visitid', 0, 'c', 0)


# ExaminationItems:
## Structure of ExaminationItems:
(0, 'ExaminationItemID', 'TEXT', 0, None, 1)
(1, 'ExaminationItem', 'TEXT', 0, None, 0)
(2, 'ReferenceValue', 'TEXT', 0, None, 0)
(3, 'Type', 'TEXT', 0, 'NULL', 0)
(4, 'Low', 'FLOAT', 0, 'NULL', 0)
(5, 'High', 'FLOAT', 0, 'NULL', 0)
(6, 'Unit', 'TEXT', 0, None, 0)
## Foreign keys of ExaminationItems:


## Indexes of ExaminationItems:
(0, 'sqlite_autoindex_ExaminationItems_1', 1, 'pk', 0)
