import { createContext, useContext } from "react";

/**
 * Whether the tab a component sits in is the one shown. Tabs stay mounted after their first
 * visit, so their forms, selections and running jobs are kept; they reload their lists when shown
 * again and pause polling while hidden. Outside a tab it is always true.
 */
export const TabActive = createContext(true);

export const useTabActive = () => useContext(TabActive);
