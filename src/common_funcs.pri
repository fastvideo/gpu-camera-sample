#
defineTest(makeDir) {
    dir = \"$$quote($$shell_path($$1))\"
    !exists($$1) {
        win32:QMAKE_POST_LINK += if not exist $$dir $${QMAKE_MKDIR} $$dir $$escape_expand(\\n\\t)
        unix: QMAKE_POST_LINK += $${QMAKE_MKDIR} -p $$dir $$escape_expand(\\n\\t)
    }
    export(QMAKE_POST_LINK)
}

# Copies the given files to the destination directory
defineTest(copyToDestdir) {
    files = $$1
    for(ifile, files) {
        exists($$ifile) {
            ofile = $${DESTDIR}/$$basename(ifile)
            !exists($$ofile) {
                ifile = \"$$quote($$shell_path($$ifile))\"
                ofile = \"$$quote($$shell_path($$ofile))\"
                QMAKE_POST_LINK += $$QMAKE_COPY $$ifile $$ofile $$escape_expand(\\n\\t)
            }
        }
        else {
           system( echo "File $$ifile does not exist !!!" )
        }
     }
    export(QMAKE_POST_LINK)
}

defineTest(copyToDestSubdir) {
    files  = $$1
    outdir = $${DESTDIR}/$$2
    makeDir( $$outdir )
    for(ifile, files) {
        ofile = $${outdir}/$$basename(ifile)
        !exists($$ofile) {
            ifile = \"$$quote($$shell_path($$ifile))\"
            ofile = \"$$quote($$shell_path($$ofile))\"
            QMAKE_POST_LINK += $$QMAKE_COPY $$ifile $$ofile $$escape_expand(\\n\\t)
         }
     }
    export(QMAKE_POST_LINK)
}

# Copies the plugins to the destination directory
defineTest(copyPluginsToDestdir) {
    idir = $$[QT_INSTALL_PLUGINS]/$$1
    odir = $${DESTDIR}/$$basename(idir)
win32 {
    makeDir($$odir)
    ifiles = $$files($${idir}/*.dll)
    for(ifile, ifiles) {
        if( greaterThan(QT_MAJOR_VERSION, 4) ) {
            d_ifile = $$find(ifile,"d.dll")
        } else {
            d_ifile = $$find(ifile,"d4.dll")
        }
        ofile = $${odir}/$$basename(ifile)
        !exists($$ofile) {
            ifile ~= s,/,\\,g
            ofile ~= s,/,\\,g
            ifile = \"$$quote($$shell_path($$ifile))\"
            ofile = \"$$quote($$shell_path($$ofile))\"
            build_pass:CONFIG(debug, debug|release) {
                !isEmpty(d_ifile): QMAKE_POST_LINK += $$QMAKE_COPY $$ifile $$ofile $$escape_expand(\\n\\t)
            }
            else:build_pass {
                isEmpty(d_ifile): QMAKE_POST_LINK += $$QMAKE_COPY $$ifile $$ofile $$escape_expand(\\n\\t)
            }
        }
    }
}
unix {
    !exists($$odir): QMAKE_POST_LINK += $$QMAKE_COPY_DIR $$idir $${DESTDIR} $$escape_expand(\\n\\t)
}
    export(QMAKE_POST_LINK)
}

# Copies the qt dlls to the destination directory
defineTest(copyQtDllsToDestdir) {
    DLLS = $$1
    for(ifile, DLLS) {
        win32 {
            CONFIG(debug, debug|release): copyToDestdir($$[QT_INSTALL_BINS]/$${ifile}"d.dll")
            CONFIG(release, debug|release): copyToDestdir($$[QT_INSTALL_BINS]/$${ifile}".dll")
        }
        unix {
            IFILE = $$ifile".so."$$QT_VERSION
            OLINK = $$ifile".so.5"
            QMAKE_POST_LINK += cd $${DESTDIR}; $$QMAKE_COPY $$[QT_INSTALL_LIBS]/$$IFILE . ; ln -sf $$IFILE $$OLINK $$escape_expand(\\n\\t)
        }
    }
    unix: export(QMAKE_POST_LINK)
}

defineTest(makeLinks) {
    unix {
        QMAKE_POST_LINK += cd $${PWD}/../../Scripts; $${PWD}/../../Scripts/make_links.sh $${DESTDIR}; $$escape_expand(\\n\\t)
        export(QMAKE_POST_LINK)
    }
}
