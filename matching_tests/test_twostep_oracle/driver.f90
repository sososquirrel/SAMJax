! test_twostep_oracle/driver.f90
!
! Stub — this test does not use a Fortran driver.
! The comparison is done entirely in Python (dump_inputs.py) by
! reading the gSAM and jsam debug dump CSV files.
!
! This file exists only for structural consistency with other tests.

program oracle_stub
  implicit none
  print *, 'test_twostep_oracle: no Fortran driver needed.'
  print *, 'Run dump_inputs.py directly for comparisons.'
end program oracle_stub
