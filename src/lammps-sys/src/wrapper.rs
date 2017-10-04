

extern "C" {
    //    pub fn lammps_open(arg1: ::std::os::raw::c_int,
    //                       arg2: *mut *mut ::std::os::raw::c_char, arg3: MPI_Comm,
    //                       arg4: *mut *mut ::std::os::raw::c_void);

    pub fn lammps_open_no_mpi(arg1: ::std::os::raw::c_int,
                              arg2: *const *mut ::std::os::raw::c_char,
                              arg3: *mut *mut ::std::os::raw::c_void);
    pub fn lammps_close(arg1: *mut ::std::os::raw::c_void);
    pub fn lammps_version(arg1: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int;
    pub fn lammps_file(arg1: *mut ::std::os::raw::c_void, arg2: *mut ::std::os::raw::c_char);
    pub fn lammps_command(arg1: *mut ::std::os::raw::c_void,
                          arg2: *mut ::std::os::raw::c_char)
                          -> *mut ::std::os::raw::c_char;
    pub fn lammps_commands_list(arg1: *mut ::std::os::raw::c_void,
                                arg2: ::std::os::raw::c_int,
                                arg3: *const *mut ::std::os::raw::c_char);
    pub fn lammps_commands_string(arg1: *mut ::std::os::raw::c_void,
                                  arg2: *mut ::std::os::raw::c_char);
    pub fn lammps_free(arg1: *mut ::std::os::raw::c_void);
    pub fn lammps_extract_setting(arg1: *mut ::std::os::raw::c_void,
                                  arg2: *mut ::std::os::raw::c_char)
                                  -> ::std::os::raw::c_int;
    pub fn lammps_extract_global(arg1: *mut ::std::os::raw::c_void,
                                 arg2: *mut ::std::os::raw::c_char)
                                 -> *mut ::std::os::raw::c_void;
    pub fn lammps_extract_box(arg1: *mut ::std::os::raw::c_void,
                              arg2: *mut f64,
                              arg3: *mut f64,
                              arg4: *mut f64,
                              arg5: *mut f64,
                              arg6: *mut f64,
                              arg7: *mut ::std::os::raw::c_int,
                              arg8: *mut ::std::os::raw::c_int);
    pub fn lammps_extract_atom(arg1: *mut ::std::os::raw::c_void,
                               arg2: *mut ::std::os::raw::c_char)
                               -> *mut ::std::os::raw::c_void;
    pub fn lammps_extract_compute(arg1: *mut ::std::os::raw::c_void,
                                  arg2: *mut ::std::os::raw::c_char,
                                  arg3: ::std::os::raw::c_int,
                                  arg4: ::std::os::raw::c_int)
                                  -> *mut ::std::os::raw::c_void;
    pub fn lammps_extract_fix(arg1: *mut ::std::os::raw::c_void,
                              arg2: *mut ::std::os::raw::c_char,
                              arg3: ::std::os::raw::c_int,
                              arg4: ::std::os::raw::c_int,
                              arg5: ::std::os::raw::c_int,
                              arg6: ::std::os::raw::c_int)
                              -> *mut ::std::os::raw::c_void;
    pub fn lammps_extract_variable(arg1: *mut ::std::os::raw::c_void,
                                   arg2: *mut ::std::os::raw::c_char,
                                   arg3: *mut ::std::os::raw::c_char)
                                   -> *mut ::std::os::raw::c_void;
    pub fn lammps_reset_box(arg1: *mut ::std::os::raw::c_void,
                            arg2: *mut f64,
                            arg3: *mut f64,
                            arg4: f64,
                            arg5: f64,
                            arg6: f64);
    pub fn lammps_set_variable(arg1: *mut ::std::os::raw::c_void,
                               arg2: *mut ::std::os::raw::c_char,
                               arg3: *mut ::std::os::raw::c_char)
                               -> ::std::os::raw::c_int;
    pub fn lammps_get_thermo(arg1: *mut ::std::os::raw::c_void,
                             arg2: *mut ::std::os::raw::c_char)
                             -> f64;
    pub fn lammps_get_natoms(arg1: *mut ::std::os::raw::c_void) -> ::std::os::raw::c_int;
    pub fn lammps_gather_atoms(arg1: *mut ::std::os::raw::c_void,
                               arg2: *mut ::std::os::raw::c_char,
                               arg3: ::std::os::raw::c_int,
                               arg4: ::std::os::raw::c_int,
                               arg5: *mut ::std::os::raw::c_void);
    pub fn lammps_scatter_atoms(arg1: *mut ::std::os::raw::c_void,
                                arg2: *mut ::std::os::raw::c_char,
                                arg3: ::std::os::raw::c_int,
                                arg4: ::std::os::raw::c_int,
                                arg5: *mut ::std::os::raw::c_void);
    pub fn lammps_create_atoms(arg1: *mut ::std::os::raw::c_void,
                               arg2: ::std::os::raw::c_int,
                               arg3: *mut ::std::os::raw::c_int,
                               arg4: *mut ::std::os::raw::c_int,
                               arg5: *mut f64,
                               arg6: *mut f64,
                               arg7: *mut ::std::os::raw::c_int,
                               arg8: ::std::os::raw::c_int);
}
